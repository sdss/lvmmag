#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-08-05
# @Filename: __main__.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import click


@click.group(name="lvmmag")
def lvmmag_cli():
    """lvmmag command line interface."""

    return


def lvmmag():
    lvmmag_cli()


if __name__ == "__main__":
    lvmmag()
