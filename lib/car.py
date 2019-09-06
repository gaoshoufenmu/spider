import os
import sys
import db
import numpy as np
from config import cfg


def sixty_mean_price(price_s):
    ps = price_s.split('-')
    if len(ps)==2:
        min_p, max_p = ps
    elif len(ps)==1:
        min_p, max_p = ps[0], ps[0]
    else:
        min_p, max_p = 0, 0
    min_p, max_p = float(min_p), float(max_p)
    sixty_p = min_p+(max_p-min_p)*0.6
    return sixty_p


class Car(object):
    """
    Stores car data and provides some useful methods
      for analyse.
    """
    def __init__(self, origin_data):
        if origin_data is None or len(origin_data) == 0:
            raise ValueError('origin_data can not be empty')
        # All category dictionary. Used for ranking
        self._cat = {}
        # save the data for subsequent processing
        self._data = {}
        # use linear regression to retrieve missing value
        #   for oil_wear
        self._reg_params = None
        # three kind of linear regression models
        self._reg_type = ['汽油', '插电式混合动力', '柴油']
        self._reg_fld = 'oil_wear'      # oil_wear idx
        # self._add_idx = len(origin_data[0])
        self._val = None
        # cache those data used for training LR model
        reg_data = {k:[] for k in self._reg_type}
        # min-max for each field. Used for normalization
        #   we do normalization in lazy-mode
        self._mms = {k:[sys.maxsize, -sys.maxsize] 
                     for k in origin_data[0]}

        

        for r in origin_data:
            # if not isinstance(r, list):
            #     raise TypeError('data row must be of type <list>')

            self._data[r['src_id']] = r         # indexing data for fast retrieving
            if 'PRICE_RANGES' in cfg:
                self._classify_b(
                    r['cat_name'],
                    r['grade'],
                    r['energy_type'],
                    r['src_id']
                )
            else:
                self._classify(r['cat_name'],
                        r['grade'],
                        r['energy_type'],
                        r['src_id'])

            # save for training oil_wear regressor in future
            if r['energy_type'] in reg_data and r['oil_wear'] > 0:
                reg_data[r['energy_type']].append(
                    (r['engine'], r['oil_wear']))
            
            # store min & max for data-normalization in future
            for k in r:
                if k == self._reg_fld:
                # for `oil_wear`, linear regressor is used to guess 
                #   missing value, so here we skip handling 
                #   of `oil_wear`
                    continue
                if isinstance(r[k], (int, float)):
                    if r[k] < 0:
                        r[k] = 0
                    if r[k]<self._mms[k][0]:
                        self._mms[k][0]=r[k]
                    if r[k]>self._mms[k][1]:
                        self._mms[k][1]=r[k]

        self.score_coeff()

        # get regression params
        if cfg.REG_PARAM == 'runtime':
            # calc regression params
            if not os.path.exists('../data/car_reg_params.npy'):
                reg_data = np.hstack(
                    (np.array(reg_data[k]) for k in self._reg_type)
                )
                self._reg(reg_data)
                np.save('../data/car_reg_params.npy', self._reg_params)
            else:
                self._reg_params = np.load('../data/car_reg_params.npy')
        elif cfg.REG_PARAM == 'pretrain':
            self._reg_params = np.array(cfg.REG_PARAM_DEF)
        else:
            raise ValueError('can not recognize the regressor param type: %s'
                % cfg.REG_PARAM
            )
        
        # guess the missing value for `oil_wear`
        for r in origin_data:
            if r['oil_wear'] <= 0 and r['energy_type'] in self._reg_type:
                w, b = self._reg_params[
                    self._reg_type.index(r['energy_type']),:]
                r['oil_wear'] = r['engine']*w + b
            # update the oil_wear minimum and maximum
            if r[self._reg_fld]>self._mms[self._reg_fld][1]:
                self._mms[self._reg_fld][1]=r[self._reg_fld]
            if r[self._reg_fld]<self._mms[self._reg_fld][0]:
                self._mms[self._reg_fld][0]=r[self._reg_fld]
                
    def _reg(self, reg_data):
        """ calculate the regressor parameters
        """
        def reg_inner(lst):
            assert a.shape[1] == 2
            x, y = a[:,0], a[:,1]
            A = np.vstack([x, np.ones(len(x))]).T
            w= np.linalg.lstsq(A, y, rcond=None)[0]
            return w

        self._reg_params = np.vstack(
            (reg_inner(reg_data[:,2*i:2*i+2]) 
             for i in range(reg_data.shape[1]/2))
        )

    def _plot_reg(self, reg_data):
        """ Plot the linear regression
        """
        if not isinstance(reg_data, np.array):
            raise TypeError("param `reg_data` must have type 'np.array', "
                "instead of '%s'" % type(reg_data))

        import matplotlib.pyplot as plt
        fig, axs = plt.subplot(nrows=1, ncols=len(reg_data.shape[1]/2), 
                               constrained_layout=True)
        i=0
        for ax in axs.flatten():
            x, y = reg_data[:,i:i+2]
            w, b = self._reg_params[i/2]
            ax.plot(x, y, 'o', label='origin data', markersize=3)
            ax.plot(x, w*x + b, 'r', label='fitted line')
            ax.set_xlabel('engine')
            ax.set_ylabel('oil_wear')
            ax.legend()
            ax.set_title(self._reg_type[i/2])
            i+=2
        plt.show()

    def _classify_b(self, cat_name, grade, energy_type, src_id):
        if not (cat_name and grade and energy_type):
            return
        c = self._cat.get(cat_name)
        if c is None:
            c = {}
            self._cat[cat_name] = c
        g = c.get(grade)
        if g is None:
            g = {}
            c[grade] = g
        e = g.get(energy_type)
        if e is None:
            e = {}
            g[energy_type] = e
        sixty_price=sixty_mean_price(
                        self._data[src_id]['guide_price'])
        for i in range(len(cfg.PRICE_RANGES)):
            if sixty_price < cfg.PRICE_RANGES[i]:
                if i == 0:
                    ps = '%d万以下' % cfg.PRICE_RANGES[i]
                else:
                    ps = '%d-%d万' % (cfg.PRICE_RANGES[i-1], cfg.PRICE_RANGES[i])
                break
        else:
            ps = '%d万以上' % cfg.PRICE_RANGES[-1]
        if ps not in e:
            e[ps]=[]
        e[ps].append(src_id)
        
        

    def _classify(self, cat_name, grade, energy_type, src_id):
        """
        Classify all data according to three indices: cat_name, grade 
          and energy_type. If the number of one kind of classified data
          is larger than 20, go on classifing it by 60%-mean-price. See
          `split` for more info.
        """
        
        
        if not (cat_name and grade and energy_type):
            return
        c = self._cat.get(cat_name)
        if c is None:
            c = {}
            self._cat[cat_name] = c
        g = c.get(grade)
        if g is None:
            g = {}
            c[grade] = g
        

        if cfg.CAT_MANUAL._SWITCH:      # cat manually
            cm = cfg.CAT_MANUAL.get(cat_name)
            # if cat_name == 'SUV' and grade.startswith('小型'):
            #     print ('fuck')
            if cm is not None:
                gm = cm.get(grade)
                if gm is not None:
                    energy_type_alias = energy_type
                    # no combination of subtype of oil now  --update at 20190409
                    if energy_type == '汽油' or energy_type == '柴油':
                        energy_type_alias = '燃油' if '汽油' not in gm else energy_type
                        # if '汽油' not in gm:    # do not distinguish the 2 kind of oils
                        #     energy_type = '燃油'
                    em = gm.get(energy_type_alias)
                    if em is not None:      # split into several subranges by price
                        assert isinstance(em, list)
                        e = g.get(energy_type)
                        if e is None:
                            e = {}
                            g[energy_type]=e
                        # match the price range
                        sixty_price=sixty_mean_price(
                            self._data[src_id]['guide_price'])
                        for i in range(len(em)):
                            if em[i]>sixty_price:
                                key = '%d万以下'%em[i] if i==0 else \
                                    '%d-%d万'%(em[i-1], em[i])
                                break
                        else:
                            key = '%d万以上'%em[-1]
                        if key not in e:
                            e[key]=[]
                        e[key].append(src_id)
                        return
        e = g.get(energy_type)
        if e is None:
            e = []
            g[energy_type] = e
        e.append(src_id)



    def split(self):
        if cfg.CAT_MANUAL._SWITCH:
            raise ValueError('Cannot split category in manual categorization mode')
        self._mean_price = {}
        for c in self._cat:                 # cat_name
            for g in self._cat[c]:          # grade
                for e in self._cat[c][g]:   # energy_type
                    ids = self._cat[c][g][e]
                    pd = {i:sixty_mean_price(self._data[i]['guide_price']) 
                          for i in ids}
                    self._mean_price.update(pd)
                    ps = list(pd.values())
                    ps.sort()       # sort ascendly
                    if (ps[-1]-ps[0]) < 2:
                        continue            # do not split
                    
                    d={}
                    if len(ids)<=10:
                        continue
                    elif len(ids)<=20:
                        n=1
                    elif len(ids)<=50:
                        n=2
                    elif len(ids) <= 70:
                        n=3
                    elif len(ids) <= 100:
                        n=4
                    else:
                        n=5
                    ss, lo = self._split(ps, n)
                    if not ss:
                        continue
                    groups=[]
                    ends=[]
                    for i in range(len(ss)):
                        if i == 0:
                            group=[iid for iid in pd if pd[iid]<ss[i]]
                            groups.append(group)
                            ends.append((None,ss[i]))
                            d['%d万以下'%ss[i]]=group
                        if len(ss)>1 and i > 0:
                            group=[iid for iid in pd if ss[i-1]<=pd[iid]<ss[i]]
                            groups.append(group)
                            ends.append((ss[i-1],ss[i]))
                            d['%d-%d万'%(ss[i-1],ss[i])]=group
                        if i == len(ss)-1:
                            group=[iid for iid in pd if pd[iid]>=ss[i]]
                            groups.append(group)
                            ends.append((ss[i],None))
                            d['%d万以上'%ss[i]]=group
                    if cfg.MERGE and len(groups)>1:
                        for i in range(len(groups)):
                            if len(groups[i]) < 5:      # merge
                                if i==0:
                                    groups[i+1]+=groups[i]
                                    groups[i]=None
                                    ends[i+1]=(ends[i][0], ends[i+1][1])
                                    ends[i]=None
                                elif i==len(groups)-1:
                                    if groups[i-1] is not None:
                                        groups[i-1]+=groups[i]
                                        groups[i]=None
                                        ends[i-1]=(ends[i-1][0], ends[i][1])
                                        ends[i]=None
                                else:
                                    if groups[i-1] is None and groups[i+1] is not None:
                                        groups[i+1]+=groups[i]
                                        groups[i]=None
                                        ends[i+1]=(ends[i][0], ends[i+1][1])
                                        ends[i]=None
                                    elif groups[i-1] is not None and groups[i+1] is not None:
                                        if len(groups[i-1]) <= len(groups[i+1]):
                                            groups[i-1]+=groups[i]
                                            ends[i-1]=(ends[i-1][0], ends[i][1])
                                        else:
                                            groups[i+1]+=groups[i]
                                            ends[i+1]=(ends[i][0], ends[i+1][1])
                                        groups[i]=None
                                        ends[i]=None
                                    elif groups[i-1] is not None and groups[i+1] is None:
                                        groups[i-1]+=groups[i]
                                        groups[i]=None
                                        ends[i-1]=(ends[i-1][0], ends[i][1])
                                        ends[i]=None    
                                
                        d={}
                        for i in range(len(groups)):
                            if groups[i] is None:
                                continue
                            st, ed = ends[i]
                            if st is None and ed is None:
                                d=groups[i]
                                break
                            elif st is None:
                                d['%d万以下'%ed]=groups[i]
                            elif ed is None:
                                d['%d万以上'%st]=groups[i]
                            else:
                                d['%d-%d万'%ends[i]]=groups[i]
                    self._cat[c][g][e]=d

    def splitn(self):
        """
        Divide by 60%-mean-price

        The concrete strategy of classifing by price is as following:
        1. 20 < number <= 50: divide into two parts
        2. 50 < number <= 75: divide into three parts
        3. 75 < number <= 100: divide into four parts
        4. 100 < number: divide into five parts

        Divide the price range to make all subrange contains the same 
          number of data as much as possible.
        """
        for c in self._cat:                 # cat_name
            for g in self._cat[c]:          # grade
                for e in self._cat[c][g]:   # energy_type
                    ids = self._cat[c][g][e]

                    if len(ids) <= 20:      # do not split by price
                        continue
                    pd = {i:sixty_mean_price(self._data[i]['guide_price']) 
                          for i in ids}
                    ps = list(pd.values())
                    ps.sort()       # sort ascendly
                    if (ps[-1]-ps[0]) < 2:
                        continue            # do not split
                    d = {}
                    if len(ids)<=50:        # split into 2 subranges
                        mv = self._split2(ps)
                        d['%d万以下'%mv]=[i for i in pd if pd[i]<mv]
                        d['%d万以上'%mv]=[i for i in pd if pd[i]>mv]
                        
                    # elif len(ids)<=75:      # split into 3 subranges
                    #     m1, m2 = self._split3(ps)
                    #     d['%d万以下'%m1]=[i for i in pd if pd[i]<m1]
                    #     d['%d万以上'%m2]=[i for i in pd if pd[i]>m2]
                    #     d['%d-%d万'%(m1,m2)]=[i for i in pd if m1<=pd[i]<=m2]

                    else:                   # split into n subranges
                        if len(ids)<=75:
                            n=3
                        elif len(ids)<=100:
                            n=4
                        else:
                            n=5
                        ss = self._splitn(ps, n)
                        for i in range(len(ss)):
                            if i == 0:
                                d['%d万以下'%ss[i]]=[iid for iid in pd if pd[iid]<ss[i]]
                            elif i == len(ss)-1:
                                d['%d万以上'%ss[i]]=[iid for iid in pd if pd[iid]>ss[i]]
                            else:
                                d['%d-%d万'%(ss[i-1],ss[i])]=[iid for iid in pd 
                                                              if ss[i-1]<=pd[iid]<=ss[i]]

                    self._cat[c][g][e]=d

    def _split2(self, ps):
        m_i = len(ps)//2
        m_v = ps[m_i] if len(ps)%2 else (ps[m_i-1]+ps[m_i])/2
        # use number multiple of 5 in priority
        m_v_ = (m_v//5)*5
        r = len([x for x in ps if x > m_v_])
        l = len(ps)-r
        b = l>r
        while(abs(r-l)>1):
            m_v_ = m_v_-1 if l>r else m_v_+1
            r = len([x for x in ps if x > m_v_])
            l = len(ps)-r
            if (b ^ (l>r)):
                break
        else:
            return m_v_
        
        return int(m_v)     # if failed, use any integer instead

    def _split3(self, ps):
        n = len(ps)//3
        step = 10
        t1 = (ps[n]//step)*step
        f1=-10
        while True:
            for i in range(len(ps)):
                if ps[i]>t1:
                    f1=i-1
                    break
            
            if 0.7<f1/n<1.2:
                m1_0=t1
                break
            elif t1-ps[n]<=0:
                t1+=step
            elif step==10:
                step=5
            elif step>3:
                step-=1
            else:
                t1-=(step//2)
                m1_0=-1
                break
            t1=(ps[n]//step)*step

        
        step2=10
        t2=(ps[2*n]//step2)*step2
        f2=-10
        while True:
            for i in range(len(ps)):
                if ps[i]>t2:
                    f2=len(ps)-i+1
                    break
            
            if 0.7<f2/n<1.2:
                m2_0=t2
                break
            elif t2-ps[n]<=0:
                t2+=step2
            elif step2==10:
                step2=5
            elif step2>3:
                step2-=1
            else:
                t2-=(step2//2)
                m2_0=-1
                break
            t2=(ps[n]//step2)*step2
            
        if m1_0==-1 and m2_0!=-1:
            m1_0 = ((t1+step//2)//step2)*step2
        elif m1_0 != -1 and m2_0 == -1:
            m2_0=((t2+step//2)//step)*step
        elif m1_0 != -1 and m2_0 != -1:
            if step != step2:
                if abs(f1/n-0.3) > abs(f2/n-0.3):
                    m1_0=(m1_0//step2)*step2
                else:
                    m2_0=(m2_0//step)*step
        else:
            # both are -1:
            #   no matter use m1_0 or m2_0 as the basestone
            #   but we still try to equal the numbers of all subranges
            m1_0, m2_0 = t1, t2
            if abs(f1/n-0.3) > abs(f2/n-0.3):
                m1_0=(m1_0//step2)*step2
            else:
                m2_0=(m2_0//step)*step
        return m1_0, m2_0
        
    def _split(self, ps, n=2):
        mam = n+2
        mim =n
        if mim < 1:
            mim = 1
        loss=sys.maxsize
        for i in range(mim, mam):
            css, closs = self._split_knn(ps, i)
            if closs < loss:
                ss=css
        return ss, loss

    def _split_knn(self, ps, n=4):
        if n==1:
            pm = ps[len(ps)//2]
            loss=0
            for p in ps:
                loss += abs(p-pm)
            return None, loss

        w = len(ps)//n
        cs = [ps[w*i] for i in range(n)]
        gs = [[] for i in range(n)]
        l=0

        while True:
            loss=0
            for p in ps:
                mv=sys.maxsize
                mi=-1
                for i in range(n):
                    if abs(p-cs[i])<mv:
                        mv = abs(p-cs[i])
                        mi=i
                gs[mi].append(p)
                loss+=mv
            if l == 30:
                break
            l+=1
            cs = [np.mean(g) for g in gs]
            gs = [[] for i in range(n)]

        sentinels = []
        num=1000
        i=0
        ridx=set()
        while i < n-1:
            for met in [100,50,20,15,10,8,5,4,2,1]:
                pre=gs[i][-1]//met
                nex=gs[i+1][0]//met
                if nex>pre:
                    if met<num:
                        num=met
                    break
            else:
                gs[i]+=gs[i+1]
                gs[i+1]=gs[i]
                ridx.add(i+1)
            i+=1
        gs = [gs[idx] for idx in range(len(gs)) if idx not in ridx]
        if len(gs)==1:
            return sentinels, loss

        for i in range(len(gs)-1):
            if num==1000:
                v=int(gs[i+1][0])
            else:
                low = (gs[i][-1]//num)*num
                high = (gs[i][0]//num)*num
                v=((low+high)/2//num)*num
                # v=(gs[i+1][0]//num)*num
            if v > 0:
                sentinels.append(v)
        
        return sentinels, loss


    def _splitn(self, ps, n=4):
        """
        Split into n subranges.

        We define a metric to measure the degree of asymmetry when splitting.
        Suppose there are n subranges, and each of them has m_i items
          respectively. Then let
          D = \sum_{i < j}^{n^2} |m_i-m_j|
          and we can see the larger D means the more asymmetric.
        
          We should refine the sentinel point untill D meets the minimum. So 
          the question is how to make D smaller after each iteration?
          One obvious method is to find the minimum, maxinum and medium of m_i, 
          then make the mininum and maxinum to be closer to medium, and repeat 
          this process untill D meets its mininum.
        """
        def D(ms):
            c = len(ms)
            d = 0
            for i in range(c):
                for j in range(i+1,c):
                    d += abs(ms[i]-ms[j])
            return d

        def Counts(ss):
            counts=[]
            c=0
            si = 0
            for i in range(len(ps)):
                if ps[i]> sentinels[si]:
                    counts.append(c)
                    c=1
                    si+=1
                    if si==len(sentinels):
                        counts.append(len(ps)-i)
                        break
                else:
                    c+=1
            else:
                counts.append(c)
            assert len(counts)==n
            return counts
        
        w = len(ps)//n

        if ps[-1]//100>1:
            step=100
        elif ps[-1]//50>1:
            step=50
        else:
            step=10

        sentinels = []
        
        for i in range(n-1):
            v = (ps[w*(i+1)]//step)*step
            if sentinels:
                if sentinels[-1] == v:
                    continue
            sentinels.append(v)
        n = len(sentinels)+1
        # if (n==2):
        #     return self._split2(ps)

        counts = Counts(sentinels)
        D0=D(counts)
        sentinelsm=sentinels[:]
        Dm=D0

        i=0
        while i<20:
            if D0<=(n-1)*(n-2)/2:
                break
            # counts_a = np.array(counts, dtype=np.int32)
            max_i = np.argmax(counts)
            min_i = np.argmin(counts)

            if max_i == n-1:
                sentinels[max_i-1]+=step
            else:
                sentinels[max_i]-=step
            if min_i == n-1:
                sentinels[min_i-1]-=step
            else:
                sentinels[min_i]+=step
            counts = Counts(sentinels)
            D1 = D(counts)

            if D1<Dm:
                Dm=D1
                sentinelsm=sentinels[:]

            if D1>D0:   # strategy useless, change `step` instead
                if step <= 4:
                    i=20
                else:
                    if step >= 10:
                        step //= 2
                    else:
                        step -= 1
                    sentinels=[(ps[w*(i+1)]//step)*step for i in range(n-1)]
                    counts=Counts(sentinels)
                    D0=D(counts)
                    i=0
                continue
            D0=D1
            i+=1
        
        return sentinelsm

    def print_cat(self):
        for c in self._cat:
            print(c)
            for g in self._cat[c]:
                gg = ' '*len(c)+'|-'+g
                print(gg)
                for e in self._cat[c][g]:
                    ee = ' '*len(gg)+'|-'+e
                    df = isinstance(self._cat[c][g][e], dict)
                    if df:
                        print(ee)
                        for p in self._cat[c][g][e]:
                            pp = ' '*len(ee)+'|-'+p
                            print(pp, len(self._cat[c][g][e][p]), 
                                  [self._mean_price[i] for i in self._cat[c][g][e][p]])
                    else:
                        print(ee, len(self._cat[c][g][e]))
        print('---', flush=True)
        print('---', flush=True)
        

    def normalize(self, ids):
        """
        Normalize data according to given ids
        @params ids: a list of id, which belongs to one cat 
        @return: numpy matrix, each row corresponds to one id
                    and columns are all nomalized numerical value 
                    with the same order of those in database
        """
        w=0
        rows=[]
        for i in ids:
            r = self._data[i]
            row = [(r[j]-self._mms[j][0])/(self._mms[j][1]-self._mms[j][0]) 
                   for j in r if isinstance(r[j], (int, float))]

            if w == 0:
                w=len(row)
            else:
                assert len(row)==w
            rows.append(row)
        return np.array(rows, dtype=np.float32)

    def score_coeff(self):
        def score_coeff_inner(d, coeff):
            if not isinstance(d, dict):
                raise TypeError('d must be a dictionary')
            if '_self' not in d:
                raise ValueError("dictionary must contains '_self'")
            coeff *= np.array(d['_self'])
            for k in d:
                if isinstance(d[k], dict):
                    score_coeff_inner(d[k], coeff)
                elif k == '_self':
                    continue
                else:
                    r = num_fld_names.index(k)
                    self._score_coeff[r,:]=d[k]*coeff

        for k in self._data:
            val = self._data[k]
            break

        num_fld_names = [k for k in val 
                           if isinstance(val[k], (int, float))]
        
        self._score_coeff = np.zeros((len(num_fld_names), 3), dtype=float)
        score_coeff_inner(cfg.SCORE_COEFF, 1)

    def score(self, ids, energy_type):
        """
        Caculate the scores of given ids
        @params ids: a list of ids which belong to one category
        @return: a matrix whose 1st column stores ids while 
                    2nd column stores corresponding scores. 
                    Note that the matrix is sorted by its 2nd column descently.
        """
        # no combination of subtype of oil now  --update at 20190409
        if energy_type == '汽油' or energy_type == '柴油' \
            or energy_type == '油电混合':  # or energy_type == '燃油' 
            i=1
        elif energy_type == '纯电动':
            i=0
        elif energy_type == '插电式混合动力':
            i=2
        else:
            raise ValueError("The model for energy_type '%s' is not available"
                % energy_type
            )

        data = self.normalize(ids)
        score = np.matmul(data, self._score_coeff[:,i]).ravel()
        idxs = np.argsort(-score)
        return np.stack((np.array(ids)[idxs], score[idxs])).T

    def _uniform_grade(self, grade):
        if grade.startswith('大型'):
            return '大型车'
        if grade.startswith('紧凑型'):
            return '紧凑型车'
        if grade.startswith('小型'):
            return '小型车'
        if grade.startswith('中大型'):
            return '中大型车'
        if grade.startswith('中型'):
            return '中型车'
        return grade

    def _prehandle(self, data, md):
        if md[1] == 'float':
            return str(data)
        if md[0] == 'grade':
            return self._uniform_grade(data)
        return data

    def rank2db(self):
        sql = "INSERT INTO CarRank (name, engine, mileag, max_power, charge_fast, " \
              "preserve_mean, guide_price, index_url, cat_name, grade, energy_type" \
              ", price_range, score, rank) VALUES ("
        sql += ', '.join([
            "%d" if f[1] == 'int' else "%s"
            for f in cfg.SHOW_FIELDS.data])
        sql += ')'

        for cat in self._cat:
            for grade in self._cat[cat]:
                for energy in self._cat[cat][grade]:
                    # if cat == 'SUV' and grade.startswith('小型'):
                    #     print ('fuck')
                    if energy not in \
                        ['汽油', '柴油', '油电混合', '纯电动', '插电式混合动力']:
                        continue
                    d = self._cat[cat][grade][energy]
                    res=[]
                    if isinstance(d, list):
                        if len(d) == 0: continue

                        m=self.score(d, energy)
                        for i in range(m.shape[0]):
                            res.append(
                                tuple(
                                    self._prehandle(self._data[m[i][0]][fld[0]], fld) \
                                        for fld in cfg.SHOW_FIELDS.data[:-3]
                                )
                                +
                                (
                                    "", 
                                    round(m[i][1],4), 
                                    i+1
                                )
                            )
                        db.mssql_execmany(sql, res)
                    else:
                        for k in d:
                            m=self.score(d[k], energy)
                            for i in range(m.shape[0]):
                                res.append(
                                    tuple(
                                        self._prehandle(self._data[m[i][0]][fld[0]], fld) \
                                            for fld in cfg.SHOW_FIELDS.data[:-3]
                                    )
                                    +
                                    (
                                        k, 
                                        str(round(m[i][1],4)),
                                        i+1
                                    )
                                )
                        # db.mssql_exec(sql%res[0])
                        db.mssql_execmany(sql, res)


    def show_score(self, ids, energy_type, f=None):
        m=self.score(ids, energy_type)
        d=self._data[ids[0]]
        if cfg.SHOW_FIELDS._SWITCH:
            ks=cfg.SHOW_FIELDS.data
            ks.append('score')
            if f:
                f.write('\t'.join(ks)+'\n')
            else:
                print('\t'.join(ks))
        else:
            ks=list(d.keys())
            ks.append('score')
            if f:
                f.write('\t'.join(ks)+'\n')
            else:
                print('\t'.join(ks))

        for i in range(m.shape[0]):
            vs = [self._data[m[i][0]][ks[j]] for j in range(len(ks)-1)]
            vs = [v if isinstance(v, str) else str(v) for v in vs]
            vs.append(str(m[i][1]))
            if f:
                f.write('\t'.join(vs)+'\n')
            else:
                print('\t'.join(vs))
