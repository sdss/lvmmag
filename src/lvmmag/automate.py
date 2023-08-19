#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-08-17
# @Filename: automate.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import multiprocessing
import pathlib
import warnings
from functools import partial

import healpy
import numpy
import rich.progress
from sdssdb.peewee.sdss5db import database

from lvmmag.filter import calculate_magnitudes
from lvmmag.query import query_healpix


__all__ = ["automate"]


def _get_fname(nside, ipix):
    """Returns the filename associated with a HEALPix pixel."""

    zpad = int(numpy.log10(healpy.nside2npix(nside))) + 1
    return f"gaia_dr3_xp_sampled_median_spectrum_{nside}_{ipix:0{zpad}}.h5"


def _query_ipix(
    nside: int,
    output_path: pathlib.Path,
    ipix: int,
    overwrite: bool = False,
    max_gmag: float | None = None,
    user: str | None = None,
    host: str | None = None,
    port: int | None = None,
):
    """Runs the HEALPix query for an ipix."""

    from sdssdb import PeeweeDatabaseConnection

    task_connection = PeeweeDatabaseConnection()
    task_connection.connect("sdss5db", user=user, host=host, port=port)

    fname = _get_fname(nside, ipix)
    ipix_path = output_path / fname

    if ipix_path.exists():
        if overwrite is False:
            return
        else:
            ipix_path.unlink()

    ipix_data = query_healpix(
        ipix,
        healpy.nside2order(nside),
        max_gmag=max_gmag,
        connection=task_connection,
    )

    sflux = numpy.array(ipix_data.flux.tolist())
    mags = calculate_magnitudes(sflux, filter_type="optimistic")

    ipix_data[["lflux", "lmag_ab", "lmag_vega"]] = mags

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        ipix_data.to_hdf(str(ipix_path), "data", mode="w", complevel=9, complib="blosc")

    return


def automate(
    order: int = 8,
    processes: int = 5,
    output_path: str | pathlib.Path = "./",
    overwrite: bool = False,
    max_gmag: float | None = None,
    user: str | None = None,
    host: str | None = None,
    port: int | None = None,
):
    """Automates the generation of LVM magnitudes.

    Parameters
    ----------
    order
        The HEALPix order value to use to tessellate the sky.
    processes
        Number of processes to use for multitasking.
    output_path
        Path where to save the intermediate files.
    overwrite
        Whether to overwrite query files. If `False` and the output file
        for an ipix exists in ``output_path``, it will be used.
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

    """

    nside = 2**order
    n_pixels = healpy.nside2npix(nside)

    if not database.connected:
        database.connect(user=user, host=host, port=port)

    if not database.connected:
        raise RuntimeError("Cannot connect to sdss5db.")

    progress = rich.progress.Progress(
        rich.progress.TextColumn("[progress.description]{task.description}"),
        rich.progress.BarColumn(bar_width=None),
        rich.progress.MofNCompleteColumn(),
        rich.progress.TimeRemainingColumn(),
        transient=False,
        expand=True,
        auto_refresh=True,
    )
    query_task = progress.add_task("[blue]Querying ...", total=n_pixels)
    progress.start()

    output_path = pathlib.Path(output_path).absolute()

    _query_ipix_partial = partial(
        _query_ipix,
        nside,
        output_path,
        overwrite=overwrite,
        max_gmag=max_gmag,
        user=user,
        host=host,
        port=port,
    )

    with multiprocessing.Pool(processes=processes) as pool:
        for _ in pool.imap(_query_ipix_partial, list(range(n_pixels))):
            progress.advance(query_task)

    progress.update(query_task, description="[green]Querying complete")
    progress.stop()
