#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-08-17
# @Filename: query.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

# pyright: reportGeneralTypeIssues=false

from __future__ import annotations

import healpy
import numpy
import pandas
from sdssdb.connection import PeeweeDatabaseConnection
from sdssdb.peewee.sdss5db import catalogdb, database


__all__ = ["cone_search", "query_healpix"]


def cone_search(
    ra: float,
    dec: float,
    radius: float,
    max_gmag: float | None = None,
    user: str | None = None,
    host: str | None = None,
    port: int | None = None,
    connection: PeeweeDatabaseConnection | None = None,
    work_mem="20GB",
):
    """Retrieves XP spectra information around RA/Dec coordinates.

    Parameters
    ----------
    ra
        Right ascension around which to query XP data.
    dec
        Declination around which to query XP data.
    radius
        The radius, in degrees, of the search cone.
    max_gmag
        If set, the maximum Gaia G magnitude for the targets to return.
    user
        The username used to connect to the database. If `None`, uses the default
        for the loaded profile. As with the other database parameters, if the
        database connection is already connected, it will not try to reconnect.
    host
        The host on which the database server is running.
    port
        The port on which the database server is serving.
    connection
        A ``PeeweeDatabaseConnection`` to use instead of the ``sdssdb`` one.
        This is useful for multiprocessing when one needs to create a different
        connection for each process.
    work_mem
        The value to which to set the PostgreSQL ``work_mem`` parameter during
        the transaction.

    Returns
    -------
    data
        A Pandas data frame with the selected sources. The data frame includes
        all the columns in ``xp_sampled_mean_spectrum`` (see
        https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_spectroscopic_tables/ssec_dm_xp_sampled_mean_spectrum.html)
        as well as ``phot_g_mean_mag``, ``phot_bp_mean_mag``, and ``phot_rp_mean_mag``.

    """

    Gaia_DR3 = catalogdb.Gaia_DR3
    Gaia_XP = catalogdb.Gaia_dr3_xp_sampled_mean_spectrum

    if connection is None:
        if not database.connected:
            database.connect(user=user, host=host, port=port)

        if not database.connected:
            raise RuntimeError("Cannot connect to sdss5db.")

        connection = database

    else:
        Gaia_DR3._meta.database = connection
        Gaia_XP._meta.database = connection

    query = (
        Gaia_DR3.select(
            Gaia_XP,
            Gaia_DR3.phot_g_mean_mag,
            Gaia_DR3.phot_bp_mean_mag,
            Gaia_DR3.phot_rp_mean_mag,
        )
        .join(
            Gaia_XP,
            on=(Gaia_XP.source_id == Gaia_DR3.source_id),
        )
        .where(Gaia_DR3.cone_search(ra, dec, radius))
    )

    if max_gmag is not None:
        query = query.where(Gaia_DR3.phot_g_mean_mag <= max_gmag)

    with connection.atomic():
        connection.execute_sql(f'SET LOCAL work_mem = "{work_mem}";')
        df = pandas.DataFrame.from_records(query.dicts())

    if len(df) == 0:
        return df

    # Convert string arrays to numpy arrays.
    for col in ["flux", "flux_error"]:
        df[col] = df[col].apply(lambda x: numpy.array(eval(x), dtype=numpy.float32))

    return df


def query_healpix(ipix: int, order: int, nest: bool = True, **cone_search_kwargs):
    """Retrieves XP spectra information for a HEALPix pixel.

    Works similarly to `.cone_search` but returns a data frame with the spectra
    for targets that belong to a given HEALPix pixel.

    This function is less efficient that `.cone_search` because it first does
    a RA/Dec cone search with a radius that includes the HEALPix pixel, then
    rejects targets not in the pixel. However, it's a convenient way to tessellate
    the entire sphere, especially for multiprocessing.

    Parameters
    ----------
    ipix
        The pixel index.
    nside
        The HEALPix order parameter.
    nest
        Whether to use nest pixel ordering or ring.
    cone_search_kwargs
        Arguments to pass to `.cone_search`.

    Returns
    -------
    data
        A Pandas data frame with the spectra for the matching targets.
        See `.cone_search` for more details.

    """

    nside = 2**order

    # Centre of the pixel.
    ra, dec = healpy.pix2ang(nside, ipix, nest=nest, lonlat=True)

    # Get the maximum "radius" of the nside pixels. We use this for the initial
    # cone search to get at least all the possible targets in the pixel.
    radius = healpy.max_pixrad(nside, degrees=True) / 2

    cone_search_kwargs.update({"ra": ra, "dec": dec, "radius": radius})
    data = cone_search(**cone_search_kwargs)

    if len(data) == 0:
        return data

    # Get the ipix for each one of the returned targets.
    data_ipix = healpy.ang2pix(nside, data.ra, data.dec, nest=nest, lonlat=True)

    # Select only those targets with matching ipix
    return data.loc[data_ipix == ipix]
