"""
Read/write database
"""


import pymssql
import functools
from config import cfg


# def mssql_cursor(func):
#     with pymssql.connect(host=cfg.DB_CONN.HOST,
#                          user=cfg.DB_CONN.USER,
#                          password=cfg.DB_CONN.PASSWORD,
#                          database=cfg.DB_CONN.DATABASE) as conn:
#         with conn.cursor() as cursor:
#             return functools.partial(func, cursor=cursor)

def mssql_exec(query):
    with pymssql.connect(host=cfg.DB_CONN.HOST,
                         user=cfg.DB_CONN.USER,
                         password=cfg.DB_CONN.PASSWORD,
                         database=cfg.DB_CONN.DATABASE) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            conn.commit()

def mssql_execmany(sql, rows):
    with pymssql.connect(host=cfg.DB_CONN.HOST,
                         user=cfg.DB_CONN.USER,
                         password=cfg.DB_CONN.PASSWORD,
                         database=cfg.DB_CONN.DATABASE) as conn:
        with conn.cursor() as cursor:
            cursor.executemany(sql, rows)
            conn.commit()


def mssql_select(query):
    with pymssql.connect(host=cfg.DB_CONN.HOST,
                         user=cfg.DB_CONN.USER,
                         password=cfg.DB_CONN.PASSWORD,
                         database=cfg.DB_CONN.DATABASE) as conn:
        with conn.cursor(as_dict=True) as cursor:
            cursor.execute(query)
            return [r for r in cursor]


def create_res_table():
    if cfg.SHOW_DEST != 'database':
        raise ValueError("'SHOW_DEST' in configuration must be 'database'")

    sql = """
        if object_id(N'dbo.CarRank', N'U') is not null
            truncate table CarRank
        else
            begin
    """
    sql += "create table CarRank (id int identity(1,1) NOT NULL primary key"

    for f in cfg.SHOW_FIELDS.data:
        sql += ", " + ' '.join(f)
    sql += "); \n\n"
    sql += """CREATE NONCLUSTERED INDEX cat_idx on CarRank (
        [cat_name] asc,
        [grade] asc,
        [energy_type] asc
    );
    end"""

    mssql_exec(sql)
