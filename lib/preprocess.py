"""
Preprocess data.

All datas are gathered from web and may be not completed. We 
  need to use some kinds of strategies to process missing values.
  Zero-padding is a common method for the purpose, but in some 
  cases, using regressor achieves better effect.

"""


from .config import cfg

def miss_val(x=None, fld=None):
    if x is None:   # default handler
        if cfg.MISS_VAL_HANDLER == 'zero_pad':
            return 0
        else:
            raise NotImplementedError(
                'No implementation for miss_val_handler: %s' %
                cfg.MISS_VAL_HANDLER
            )
    else if fld is not None:           # using regressor
        mth_name = cfg.REG_TYPE + fld
        return globals()[mth_name](x)

    else:
        raise ValueError(
            'param `fld` can not be None when x is not None'
        )
    
        
def linear_oil_wear(x):
    pass


def classify(rs):
    """
    Classify all data according to three indices: cat_name, grade 
      and energy_type. If the number of one kind of classified data
      is larger than 20, go on classifing it by 60%-mean-price.
    
    
    """
    cat={}

    # firstly, classify data according to 3 fixed and necessify indices
    for r in rs:    
        if r['cat_name'] in cat:
        else:
            cat[r['cat_name']] = {r['grade']:}