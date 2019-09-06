# -*- coding: utf-8 -*-
import argparse
import sys
import os
import _init_paths
import db
from car import Car
from config import cfg
import chardet
# from config import cfg_from_file

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='rank cars')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--cat', dest='cat',
                        help='category name',
                        default=None, type=str)
    parser.add_argument('--grade', dest='grade',
                        help='grade name',
                        default=None, type=str)
    parser.add_argument('--energy', dest='energy',
                        help='energy name',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    args = parser.parse_args()
    return args

def rank(c, cat, grade, energy):
    d = c._cat[cat][grade][energy]
    if cfg.SHOW_DEST == 'file':
        dpath = os.path.dirname(__file__) + '/../data/genfile/'
        if not os.path.exists(dpath):
            os.makedirs(dpath)
    if isinstance(d, list):
        try:
            if cfg.SHOW_DEST == 'file':
                p = dpath+cat+'_'+grade+'_'+energy+'.txt'
                f = open(
                    p,
                    'w', encoding='utf-8')
            else:
                f = None
                print('-'*10, cat, grade, energy, '-'*10)
        except Exception as e:
            print(e)
            raise e
        c.show_score(d, energy, f)
        if f:
            f.close()
    else:
        for k in d:
            try:
                if cfg.SHOW_DEST == 'file':
                    if '-' in k:
                        kf = k[:-1]
                    elif '上' in k:
                        kf = 'ge'+k[:-3]
                    else:
                        kf = 'lt'+k[:-3]
                    p = dpath +cat +'_'+grade+'_'+energy+kf+'.txt'
                    f = open(p, 'w', encoding='utf-8')
                else:
                    f = None
                    print('-'*10, cat, grade, energy, '-'*10)
            except Exception as e:
                print(e)
                raise e
            if not f:
                print('-'*20, k, '-'*20)
            c.show_score(d[k], energy, f)
            if not f:
                print('='*50)
            else:
                f.close()


def main():
    args = parse_args()
    c = Car(db.mssql_select('select * from CarData'))

    if cfg.SHOW_DEST == 'database':
        db.create_res_table()
        # rank all and then store results in database
        c.rank2db()
        return

    if not cfg.CAT_MANUAL._SWITCH:
        c.split()
        # c.print_cat()

    ae=args.energy.encode('GBK').decode('utf-8') if args.energy else None
    
    # no combination of subtype of oil now  --update at 20190409
    # if not cfg.CAT_MANUAL._SWITCH:
    #     if ae == '燃油':
    #         ae = '汽油'
    #         print('燃油 is be replaced by 汽油 in auto-category mode')

    ts = []
    ac = args.cat.encode('GBK').decode('utf-8') if args.cat else None
    ag = args.grade.encode('GBK').decode('utf-8') if args.grade else None
    for cat in c._cat:
        for grade in c._cat[cat]:
            for energy in c._cat[cat][grade]:
                if ae and energy != ae:
                    continue
                if ac and cat != ac:
                    continue
                if ag and grade != ag:
                    continue
                
                ts.append((cat, grade, energy))
    
    for t in ts:
        rank(c, *t)


if __name__ == '__main__':
    main()
