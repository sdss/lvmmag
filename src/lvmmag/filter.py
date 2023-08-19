#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-08-18
# @Filename: filter.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

from typing import cast

import astropy.io.ascii
import astropy.units
import numpy
from pyphot import Filter, unit


__all__ = ["get_filter", "calculate_magnitudes"]


def get_filter(filter_type: str = "mean") -> Filter:
    """Returns a ``pyphot`` filter for the LVM AG filter passband.

    Parameters
    ----------
    filter_type
        Either ``pessimistic``, ``optimistic``, or ``mean``. See
        Section 4 in LVM-0059. ``mean`` uses the average between optimistic
        and pessimistic.

    """

    cwd = pathlib.Path(__file__).parent

    data = astropy.io.ascii.read(
        str(cwd / "data" / "OptPess_Apr20.txt"),
        names=["wave", "optimistic", "pessimistic"],
    )

    wave = data["wave"].value * 10

    if filter_type in ["optimistic", "pessimistic"]:
        transmit = data[filter_type].value
    elif filter_type == "mean":
        transmit = 0.5 * (data["optimistic"] + data["pessimistic"]).value
    else:
        raise ValueError("Invalid filter type.")

    return Filter(
        wavelength=wave * unit["AA"],
        transmit=transmit,
        name=filter_type,
        dtype="photon",
    )


def calculate_magnitudes(
    sflux: numpy.ndarray | astropy.units.Quantity,
    wave: numpy.ndarray | astropy.units.Quantity | None = None,
    filter_type: str = "mean",
    flux_units="W m-2 nm-1",
):
    """Returns the magnitudes associated with a mean spectrum using the LVM filter.

    Parameters
    ----------
    sflux
        The spectrum flux density. It can be a 1- or 2-D array (in the latter case
        each spectrum must be a row) or an astropy quantity.
    wave
        The wavelength solution for the provided spectrum. Either a 1D array or
        an astropy quantity. If not a quantity, the units are assumed to be
        Angstrom. If not provided, the default wavelength sampling for
        ``xp_sampled_mean_spectrum`` (3360 to 10200 A with a step of 20 A)
        will be used.
    filter_type
        Either ``pessimistic``, ``optimistic``, or ``mean``. See
        Section 4 in LVM-0059. ``mean`` uses the average between optimistic
        and pessimistic.
    flux_units
        The units of the flux density. If ``flux`` is a quantity with units,
        this parameter is ignored.

    Returns
    -------
    magnitudes
        A 2D array with the flux, AB, and Vega synthetic magnitudes
        for each spectrum through the bandpass of the LVM AG cameras.
        The flux is returned in units of W m-2 nm-1. Note that even if
        the input flux was a 1D array (single spectrum), the output is
        always a 2D array.

    """

    sflux = numpy.atleast_2d(sflux)

    if not isinstance(sflux, astropy.units.Quantity):
        sflux *= astropy.units.Unit(flux_units)

    assert isinstance(sflux, astropy.units.Quantity)

    # pyphot expects erg / s / cm^2 / A
    # 1 W m-2 nm-1 = 100 erg s-1 cm-2 A-1
    sflux = cast(astropy.units.Quantity, sflux.to("erg s-1 cm-2 AA-1"))

    if wave is None:
        wave = numpy.arange(3360, 10200 + 20, 20)

    if not isinstance(wave, astropy.units.Quantity):
        wave *= astropy.units.Unit("AA")

    assert isinstance(wave, astropy.units.Quantity)

    wave = cast(astropy.units.Quantity, wave.to("AA"))

    if wave.size != sflux.shape[1]:
        raise ValueError("Wavelength sampling does not match flux.")

    filter = get_filter(filter_type)

    wave_pyphot = wave.value * unit["AA"]
    sflux_pyphot = sflux.value * unit["erg/s/cm**2/AA"]

    flux = filter.get_flux(wave_pyphot, sflux_pyphot, axis=1)

    mag_vega = -2.5 * numpy.log10(flux) - filter.Vega_zero_mag
    mag_ab = -2.5 * numpy.log10(flux) - filter.AB_zero_mag

    flux_W = flux / 100  # Convert flux to W m-2 nm-1

    return numpy.vstack((flux_W, mag_ab, mag_vega)).T
