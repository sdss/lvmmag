#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-08-19
# @Filename: ingest.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import multiprocessing
import pathlib
import warnings
from functools import partial
from io import StringIO

import pandas
import peewee
import rich.progress


__all__ = ["ingest"]


COLUMNS = ["source_id", "ra", "dec", "lflux", "lmag_ab", "lmag_vega"]


def _get_connection(
    table_name: str = "lvm_magnitude",
    schema: str = "catalogdb",
    dbname: str = "sdss5db",
    user: str | None = None,
    host: str | None = None,
    port: int | None = None,
    check_table: bool = False,
):
    """Checks that the connection to the DB can be established and the table exists."""

    try:
        conn = peewee.PostgresqlDatabase(dbname, user=user, host=host, port=port)
    except Exception as err:
        raise ValueError(f"Cannot connect to database: {err}")

    if not conn.connect():
        raise ValueError("Cannot connect to database.")

    if check_table:
        if not conn.table_exists(table_name, schema):
            raise ValueError(f"Table {schema}.{table_name} does not exist in {dbname}.")

    return conn


def _load_file(
    file_: str | pathlib.Path,
    table_name: str = "lvm_magnitude",
    schema: str = "catalogdb",
    dbname: str = "sdss5db",
    user: str | None = None,
    host: str | None = None,
    port: int | None = None,
):
    """Loads a file to the database."""

    conn = _get_connection(
        table_name=table_name,
        schema=schema,
        dbname=dbname,
        user=user,
        host=host,
        port=port,
        check_table=False,
    )

    file_ = pathlib.Path(file_)
    if not file_.exists():
        warnings.warn(f"File {file_!s} not found.", UserWarning)
        return

    data = pandas.read_hdf(file_)
    data = data.loc[:, COLUMNS]

    cursor = conn.cursor()

    try:
        with StringIO() as ss:
            data.to_csv(ss, index=False)
            ss.seek(0)
            cursor.copy_expert(f"COPY {schema}.{table_name} FROM STDIN cSV HEADER", ss)
    except Exception as err:
        warnings.warn(f"Failed copying file {file_!s}: {err}")
    finally:
        conn.commit()


def ingest(
    path: str | pathlib.Path | None = None,
    pattern: str = "*",
    files: list[pathlib.Path | str] | None = None,
    processes: int = 1,
    table_name: str = "lvm_magnitude",
    schema: str = "catalogdb",
    dbname: str = "sdss5db",
    user: str | None = None,
    host: str | None = None,
    port: int | None = None,
):
    """Loads a list of processed files with LVM magnitude information into the DB.

    Parameters
    ----------
    path
        The path where the data files can be found. The files are usually generated
        using `.automate`.
    pattern
        The regex pattern to select files from ``path``. Defaults to all the files.
    files
        Alternatively to providing a ``path``, a list of files to process can
        be passed.
    processes
        Number of processes to use to load the data.
    table_name
        The table into which to load the data. It must already exist.
    schema
        The schema in which the table lives.
    dbname
        The database name.
    user
        The username used to connect to the database.
    host
        The host on which the database server is running.
    port
        The port on which the database server is serving.

    """

    # Check the connection.
    _get_connection(
        table_name=table_name,
        schema=schema,
        dbname=dbname,
        user=user,
        host=host,
        port=port,
        check_table=True,
    )

    if files is None and path is None:
        raise ValueError("Either files or path must be provided.")

    if files is None:
        assert path is not None
        files = list(pathlib.Path(path).glob(pattern))

    if len(files) == 0:
        raise ValueError("No files found.")

    _load_file_partial = partial(
        _load_file,
        table_name=table_name,
        schema=schema,
        dbname=dbname,
        user=user,
        host=host,
        port=port,
    )

    progress_args = (
        rich.progress.TextColumn("[progress.description]{task.description}"),
        rich.progress.BarColumn(bar_width=None),
        rich.progress.MofNCompleteColumn(),
        rich.progress.TimeRemainingColumn(),
    )

    with rich.progress.Progress(
        *progress_args,
        transient=False,
        expand=True,
        auto_refresh=True,
    ) as progress:
        p_task = progress.add_task("[blue]Ingesting data ...", total=len(files))

        with multiprocessing.Pool(processes=processes) as pool:
            for _ in pool.imap_unordered(_load_file_partial, files):
                progress.advance(p_task)

        progress.update(p_task, description="[green]Ingestion complete")
