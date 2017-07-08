# -*- coding: utf-8 -*-
"""
Created on Mon May 08 13:38:44 2017

@author: Jack
"""
from __future__ import division

import numpy as np
import pandas as pd

import scipy.optimize as spo
import scipy.stats as scs
import statsmodels.api as sm

import matplotlib.pylab as plt
import matplotlib
from pylab import mpl
from imp import reload
#import mysql.connector

matplotlib.rcParams['lines.linewidth'] = 2
zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simkai.ttf')#自定义字体为楷体
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

def data_download(codelist=None,startdate=None,enddate=None,fields=None,path_name=None,ifsave=None):
    import WindPy
    WindPy.w.start()
    if codelist is None:
        codelist = ['881001.WI','000016.SH','000903.SH','000300.SH','000905.SH','000852.SH','399101.SZ','399006.SZ',
                    'SPX.GI','NDAQ.O','HSI.HI','HSCEI.HI',
                    'H11001.CSI','037.CS','000012.SH','000013.SH','H11008.CSI',
                    'CL.NYM','AU9999.SGE','AUFI.WI','AGFI.WI','SPTAUUSDOZ.IDC',
                    'NH0100.NHF','NH0200.NHF','NH0300.NHF','NH0400.NHF',
                    '052.CS']
    if startdate is None: startdate = '2000-01-01'
    if enddate is None: enddate = '2017-06-01'
    if fields is None: fields = ['close']
    if path_name is None: path_name = ['assetallocation.h5','indexdata']
    data = WindPy.w.wsd(codelist,fields,startdate,enddate)
    nd = np.array(data.Data).T
    data = pd.DataFrame(nd,index=data.Times,columns=codelist)
    if ifsave:
        data.to_hdf(path_name[0],path_name[1])
    return data

def get_index_offset(df,index,offset):
    #给定dataframe的df，和其中的某个索引index
    #返回与index相差offset行的索引，offset为正表示向下
    if not index in df.index:
        raise ValueError('index is not in df.index')
        return
    try:
        return df.index[list(df.index).index(index)+offset]
    except:
        raise ValueError('offset is too long')
        return

def nv_normalize(df):
    #传入价格dataframe,返回归一化的数据，即第一行都设为1
    df_tmp = df.copy()
    for col in df_tmp.columns:
        df_tmp[col] = df_tmp[col]/df_tmp[col][0]
    return df_tmp

def get_change_list(index,freq='q'):
    #输入某个dataframe的日期时间戳index和调仓周期
    #y,q,m,d分别表示年、季、月、周、日，将获取相应周期的最后一个交易日列表
    #返回需要的调仓日期的列表list
    i = pd.Series(index)
    if freq=='q':
        change_list = i.groupby([index.year,index.quarter]).last().tolist()
    elif freq=='m':
        change_list = i.groupby([index.year,index.month]).last().tolist()
    elif freq=='d':
        change_list = i.groupby([index.year,index.day]).last().tolist()
    elif freq=='w':
        change_list = i.groupby([index.year,index.week]).last().tolist()
    elif freq=='y':
        change_list = i.groupby([index.year]).last().tolist()
    else:
        raise ValueError('invalid freq')
    return sorted(change_list)

class stat:
    
    def __init__(self,ret,freq='d',rf=None,benchmark=None):
        '''
            the ret must be a dataframe with columns as assets, it's recommended 
            that the index of ret is timestamp
             
            freq is 'd','w','m','q','y' denoting day, week, month, quarter, 
            year separately
        
            the rf must be an annual number like 0.03 or a time series with 
            same freqency as ret, and with more observations than ret 
            
            the benchmark must be return data with same frequency as ret, 
            and with more observations than ret
            
        '''
        units_dic = {'d':250,'w':50,'m':12,'q':4,'y':1}
        self.units = units_dic[freq]#units denote how many trading units in one year
        self.obs = len(ret)#number of observations
        
        try:
            benchmark = benchmark.loc[ret.index]
        except:
            benchmark = None
        '''
            rf has 3 input types:
                1. None
                    rf = 0, rf_annual = 0
                2. annual number like 0.03
                    rf = 0.03, rf_annual = 0.03
                3. series with timestamp as index
                    rf = rf.loc[ret.index] or 0, rf_annual = annualize(rf)
        '''
        if rf is None:
            rf, rf_annual = 0, 0
        elif isinstance(rf,(float,int)):
            rf, rf_annual = np.exp(rf/self.units)-1, rf
        else:
            try:#type3
                rf = rf.loc[ret.index]
                rf_annual = (((1+rf).cumprod())[-1])**(self.units/self.obs)-1#annualize
            except:
                rf, rf_annual = 0, 0
        
        self.ret = ret
        self.n = len(ret.columns)#number of assets
        self.freq = freq
        self.rf = rf
        self.rf_annual = rf_annual
        self.benchmark = benchmark
        self.stat,self.net_value,self.drawdown,self.relative_ret = self.get_stat()
        if self.benchmark is not None:
            self.relative_nv = (self.relative_ret+1).cumprod()   
    
    def get_stat(self):
        '''
            calculate annual_return, annual_std, sharpe, sortino..., 
            alpha, information ratio...
        '''
        ret = self.ret
        rf = self.rf
        rf_annual = self.rf_annual
        benchmark = self.benchmark
        obs = self.obs
        units = self.units
        stat = {}
        annual_ret = {}
        annual_std = {}
        downside_std = {}
        total_ret = {}
        sharpe = {}
        sortino = {}
        calmar = {}
        max_drawdown = {}
        max_dd_date = {}#最大回撤对应的日期
        net_value = {}
        drawdown = {}
        
        alpha = {}#annual alpha
        beta = {}
        omega = {}#annual std of residual
        IR = {}#infomation ratio
        
        for col in ret.columns:
            ts = ret[col]
            try:
                nv = (ts+1).cumprod()
                dd = nv/(nv.cummax())-1
                max_drawdown[col] = dd.min()
                max_dd_date[col] = dd.idxmin()
                total_ret[col] = nv[-1]-1
                net_value[col] = nv
                drawdown[col] = dd
            except:
                raise ValueError('inf in total_ret')
            annual_ret[col] = (1+total_ret[col])**(units/obs)-1
            annual_std[col] = ts.std()*np.sqrt(units)
            downside_std[col] = ts[ts<0].std()*np.sqrt(units)
            try:
                sharpe[col] = (annual_ret[col]-rf_annual)/annual_std[col]
            except:
                sharpe[col] = 0
            try:
                sortino[col] = (annual_ret[col]-rf_annual)/downside_std[col]
            except:
                sortino[col] = 0
            try:
                calmar[col] = annual_ret[col]/abs(max_drawdown[col])
            except:
                calmar[col] = 0
                
            if benchmark is not None:
                y = ts - rf
                x = benchmark - rf
                est = sm.OLS(y,sm.add_constant(x)).fit()
                alpha[col] = np.exp(est.params['const']*units) - 1
                beta[col] = est.params.iloc[1]
                omega[col] = est.resid.std()*np.sqrt(units)
                try:
                    IR[col] = alpha[col]/omega[col]
                except:
                    IR[col] = 0
            
        drawdown = pd.DataFrame(drawdown,index=ret.index)    
        stat['annual_ret'] = annual_ret
        stat['annual_std'] = annual_std
        stat['total_ret'] = total_ret
        stat['sharpe'] = sharpe
        stat['sortino'] = sortino
        stat['calmar'] = calmar
        stat['max_drawdown'] = max_drawdown
        stat['max_dd_date'] = max_dd_date
        relative_ret = None
        if benchmark is not None:
            relative_ret = ret.apply(lambda x:x-benchmark.iloc[:,0])
            stat['alpha'] = alpha
            stat['beta'] = beta
            stat['omega'] = omega
            stat['IR'] = IR
        stat = pd.DataFrame(stat)
        net_value = pd.DataFrame(net_value,index=ret.index).loc[:,ret.columns]
        try:
            stat = stat.loc[ret.columns,['annual_ret','annual_std','total_ret',
                                         'sharpe','sortino','calmar','max_drawdown',
                                         'max_dd_date','alpha','beta','omega','IR']]
        except:
            stat = stat.loc[ret.columns,['annual_ret','annual_std','total_ret',
                                         'sharpe','sortino','calmar','max_drawdown',
                                         'max_dd_date']]
        return stat,net_value,drawdown,relative_ret
    
    
    def plot_nv(self):
        '''
            plot net_value and relative net_value
        '''
        ax1 = self.net_value.plot()
        try:
            ax2 = self.relative_nv.plot()
            return ax1,ax2
        except:
            return ax1,None
        
    def stat_by_groups(self,ifplot=False):
        '''
            calculate total_ret by groups like year, month, week, etc...
            
            the index of self.ret must be timestamp
            ifplot contols whether plot the bar
        '''
        def total_ret(ret):
            return ((1+ret).cumprod()-1).iloc[-1]
            
        ret = self.ret
        yearly_ret = ret.groupby([ret.index.year]).apply(total_ret)
        monthly_ret = ret.groupby([ret.index.year,ret.index.month]).apply(total_ret)
        
        if ifplot:
            def autolabel(rects):
                ''' 
                    attach some text labels
                '''
                for rect in rects:
                    height = rect.get_height()
                    plt.text(rect.get_x()+rect.get_width()/2.,1.05*height,
                                 '%.2f%%'%(height),ha='center',va='bottom')
            
            def plot_groups(ts)
                f = plt.figure()
                ax1 = f.add_subplot(2,1,1)
                rec1 = ax1.bar(np.arange(len(ts),),ts.values,color='lightseagreen')
                autolabel(rec1)
                _ = ax1.set_xticklabels([str(t) for t in ts.index])
        
        
        
        
codelist = ['881001.WI','037.CS','SPTAUUSDOZ.IDC','HSI.HI']
data = pd.read_hdf('assetallocation.h5','indexdata')[codelist+['052.CS']].dropna()
ret = data.pct_change()[1:].dropna()
rf = ret['052.CS']
ret = ret[codelist]
ret = ret.loc['2005-9':'2016-9',:]
s = Stat(ret,freq='d',rf=rf)



class PortfolioOptimization:
    def __init__(self,ret):
        self.ret = ret
        self.change_date = ret.index[-1]
        self.n = len(ret.columns)#资产数量
    
    def get_h(self,method,h0=None,h_risk=None,h_weight=None,h_given=None,RVC=None,
              ES=None,hrvc=None,**kwargs):
        #h0初始值，h_risk风险权重(只有method='erc'时需要)
        #h_weight给定的常数权重(只有method='given'时需要)，都是(n,)的ndarray
        
        if method=='EqualWeight':#等权重组合
            self.h = np.ones(self.n,)/self.n
        elif method=='GivenWeight':#给定权重组合
            self.h = h_weight
        elif method=='RiskParity':#风险平价组合
            self.h = self.__get_h_RiskParity(h0,h_risk)
        elif method=='MostDiversified':#最大分散化组合
            self.h = self.__get_h_MostDiversified(h0)
        elif method=='MinimumVariance':
            self.h = self.__get_h_MinimumVariance()
        elif method=='MeanVariance':
            self.h = self.__get_h_MeanVariance()
        elif method=='VolatilityRecip':
            self.h = self.__get_h_VolatilityRecip()
        elif method=='FixedWeight':
            if h_given is None:
                h_given = np.ones(self.n,)/self.n
            h_given = h_given/h_given.sum()
            self.h = h_given
        else:#支持自定义函数，传入method为自定义函数的函数名，other是自定义函数的其他参数
            #会吧self.ret连同other传入
            try:
                self.h = method(self.ret,**kwargs)
            except:
                raise ValueError('Invalid method: '+getattr(method,'__name__'))
                return
        other = {}
        h_es = np.ones(self.n,)/self.n
        if ES:
            h_es,other['ES'] = self.__get_h_ES(**kwargs)
            self.h = self.h*h_es*self.n
            
        h_rvc = np.ones(self.n,)/self.n
        if RVC:
            h_rvc = self.RVC(hrvc)
            self.h = h_rvc*self.h*self.n
            
        if self.h.sum()>1:
            self.h = self.h/self.h.sum()
            
            
        return self.h,h_rvc,h_es,pd.DataFrame(other)

        
    def RVC(self,hrvc=None):
        if hrvc is None:
            hrvc = np.array([1,0.5,0.5])
        factors = pd.DataFrame()
        factors['Return'] = (self.ret+1).prod()
        factors['Volatility'] = self.ret.std()
        factors['Correlation'] = self.ret.corr().sum()
        factors[1] = factors.Return.rank()
        factors[2] = factors.Volatility.rank(ascending=False)
        factors[3] = factors.Correlation.rank(ascending=False)
        score = (hrvc[0]*factors[1]+hrvc[1]*factors[2]+hrvc[2]*factors[3]).rank().values
#        score = factors[2].rank().values
        w2 = score/score.sum()
        return w2
        
    def __get_h_RiskParity(self,h0=None,h_risk=None):
        #传入资产的收益率数据dataframe，返回有约束的risk_parity的权重(n,)
        def func_min(h,V,times):
            #用于minimize
            #h是(n,)的ndarray,V是n*n的matrix
            n = len(h)
            beta = (np.array(V).dot(h)).T
            k = h_risk
            sum = 0
            for i in range(n):
                for j in range(n):
                    sum += (h[i]*beta[i]/k[i]-h[j]*beta[j]/k[j])**2
            return sum*times
        #协方差矩阵
        V = np.matrix(self.ret.cov())
        #初始值设定为等权重
        if h0 is None:h0 = np.ones(self.n,)/self.n
        if h_risk is None:h_risk = np.ones(self.n,)/self.n
        #求出权重h是(n,)的ndarray
        bnds = [(0,1)]*self.n
        cons = ({'type':'eq','fun':lambda h:h.sum()-1})
        if func_min(h0,V,1)==0:
            return h0
        times0 = 1/func_min(h0,V,1) #保证func_min返回值不至于过大或过小
        options = {'disp': False, 'iprint': 1, 'eps': 1e-10,'maxiter': 100, 'ftol': 1e-15}
        h = spo.minimize(func_min,h0,(V,times0),method='SLSQP',constraints=cons,bounds=bnds,options=options)
        h = h.x.reshape(self.n,)
        return h
    
    def __get_h_MostDiversified(self,h0=None):
        #传入资产的收益率数据dataframe，返回mdp的权重h
        def func_min(h0,V):
            #用于minimize
            #h是(n,)的ndarray,V是n*n的matrix
            h = np.matrix(h0).reshape((len(h0),1))
            sigma = np.matrix([np.sqrt(V[i,i]) for i in range(len(V))]).reshape((len(h0),1))
            return -(h.T*sigma)/((h.T)*V*h)
        #协方差矩阵
        V = np.matrix(self.ret.cov())
        #初始值设定为等权重
        if h0 is None:h0 = np.ones(self.n,)/self.n
        bnds = [(0,1)]*self.n
        cons = ({'type':'eq','fun':lambda h:h.sum()-1})
        h = spo.minimize(func_min,h0,V,method='SLSQP',constraints=cons,bounds=bnds)
        h = h.x.reshape(self.n,)
        return h
    
    def __get_h_MinimumVariance(self):
        #传入资产的收益率数据dataframe，返回mean-variance(mv)的权重h
        V = np.matrix(self.ret.cov())
        e = np.ones((self.n,1))
        h = V.I*e
        h = h/h.sum()
        return np.array(h).reshape(self.n,)
    
    def __get_h_MeanVariance(self):
        V = np.matrix(self.ret.cov())
        f = (1+self.ret).cumprod().iloc[-1]-1-0.03
        h = V.I.dot(f)
        h = h/h.sum()
        return np.array(h).reshape(self.n,)
    
    def __get_h_VolatilityRecip(self):
        #波动率倒数权重
        try:
            h = 1/self.ret.std()
            h = h/h.sum()
        except:
            h = np.ones(self.n,)/self.n
        return h
    
    def get_ES(self,ret=None,alpha=0.05,normed=False,fit=False):
        if ret is None:
            ret = self.ret
        if normed:
            norm = scs.norm(ret.mean(),ret.std())
            VaR = pd.Series(norm.ppf(alpha),index=ret.columns)
            ES = ret.mean()*20-ret.std()*np.sqrt(20)*scs.norm.pdf(scs.norm.ppf(alpha))/alpha
        else:
            VaR = ret.quantile(alpha)
            ES = ret[ret<VaR].mean()*np.sqrt(20)+(20-np.sqrt(20))*ret.mean()
        if fit:
        #正态分布拟合
            try:
                norm = scs.norm(ret.mean(),ret.std())
                x = np.linspace(norm.ppf(0.0001),norm.ppf(0.9999),1000)
                plt.plot(x,norm.pdf(x))
                plt.hist(ret.values,normed=True,bins=50)
            except:
                raise ValueError('Too many input assets')
        return VaR,ES
        
    def __get_h_ES(self,lagh=None,alpha=0.05,normed=False,EST=-0.08):
        VaR5,ES5 = self.get_ES(ret=pd.DataFrame(self.ret),normed=normed,alpha=0.05)
        ESC = ES5
        w = (EST/ESC).values
        w = np.array(list(map(lambda x:min(x,1) if x>0 else 1,w)))/self.n
        return w,ESC
        

    
class AssetAllocation:
    def __init__(self,ret,freq=None,freq_len=None,rf=None,method_list=None,
                 leverage=None,h_risk=None,l_list=None,normed=None,RVC=None,ES=None,
                 EST=None,ifcash=None,cost=0,h_given=None,hrvc=None,
                 startdate='2005-09',enddate='2016-09',
                 change_list=None):
        #h_risk只对RiskParity有效
        self.n = len(ret.columns)#资产数量
        if freq is None: freq = 'q'
        if freq_len is None: freq_len = 4
        if rf is None:
            rf = pd.Series(0,index=ret.index)
        elif isinstance(rf,float):
            rf = (rf+1)**(1/250)-1
            rf = pd.Series(rf,index=ret.index)
        else:
            try:
                rf = rf[ret.index]
            except:
                rf = pd.Series(0,index=ret.index)
        if method_list is None: method_list = ['RiskParity']
        #杠杆倍数
        if leverage is None: leverage = 1
        #资产的风险预算，一定要跟ret列资产名称的顺序对应
        if h_risk is None: h_risk = np.ones(self.n,)/self.n
        #要加杠杆的债券列表
        if l_list is None: l_list=['H11001.CSI','037.CS','000012.SH','000013.SH','H11008.CSI']
        
        self.ret = ret
        self.freq = freq
        self.freq_len = freq_len
        self.rf = rf
        self.method_list = method_list
        self.leverage = leverage
        self.h_risk = h_risk
        self.l_list = l_list
        self.n = len(self.ret.columns)
        if normed is None:
            self.normed = False
        else:
            self.normed = normed
 
        self.ret = self.__get_ret_leverage()
        self.ret_stat,self.ret_nv, self.ret_dd = Stat(ret=self.ret['2005-09':'2016-09'],rf=self.rf).get_stat()
        self.RVC = RVC
        self.ES = ES
        self.EST = EST
        self.ifcash = ifcash
        self.cost = cost
        self.h_given = h_given
        self.hrvc = hrvc
        self.startdate = startdate
        self.enddate = enddate
        self.change_list = change_list
        
        self.CODE_DICT = {'881001.WI':'万得全A',
                          '000016.SH':'上证50',
                          '000300.SH':'沪深300',
                          '000903.SH':'中证100',
                          '000905.SH':'中证500',
                          '000852.SH':'中证1000',
                          '399101.SZ':'中小板综',
                          '399006.SZ':'创业板',
                          'SPX.GI':'标普500',
                          'NDAQ.O':'纳斯达克',
                          'HSI.HI':'恒生指数',
                          'HSCEI.HI':'恒生国企指数',
                          'H11001.CSI':'中证全债',
                          '037.CS':'中债总财富(总值)',
                          '000012.SH':'上证国债',
                          '000013.SH':'上证企业债',
                          'H11008.CSI':'中证企业债(全价)',
                          'CL.NYM':'NYMEX原油',
                          'AU9999.SGE':'SGE黄金',
                          'AUFI.WI':'沪金指数',
                          'SPTAUUSDOZ.IDC':'伦敦金现',
                          'AGFI.WI':'沪银指数',
                          'NH0100.NHF':'南华商品',
                          'NH0200.NHF':'南华工业品',
                          'NH0300.NHF':'南华农产品',
                          'NH0400.NHF':'南华金属',
                          '052.CS':'央票总财富(总值,无风险)',
                          'Cash':'现金'}
        self.METHOD_DICT = {'RiskParity':'风险平价',
                            'MostDiversified':'最大分散组合',
                            'MinimumVariance':'最小方差组合',
                            'EqualWeight':'等权重组合',
                            'VolatilityRecip':'波动率倒数',
                            'FixedWeight':'固定权重',
                            'MeanVariance':'均值方差'}

        
    def get_hh(self,ifplot=False):
        h = {}
        h_rvc = {}
        h_es = {}
        others = {}
        if self.change_list is None:
            self.change_list = get_change_list(self.ret.index,self.freq)
        for change_date in self.change_list[self.freq_len:]:
            df = self.get_change_data(change_date)
            h_tmp = {}
            _,h_rvc[change_date],h_es[change_date],others[change_date] = PortfolioOptimization(df).get_h('EqualWeight',normed=self.normed,RVC=self.RVC,ES=self.ES,EST=self.EST,hrvc=self.hrvc)
            for method in self.method_list:
                h_tmp[method],_,_,_ = PortfolioOptimization(df).get_h(method,h_risk=self.h_risk,normed=self.normed,
                           RVC=self.RVC,ES=self.ES,EST=self.EST,
                           h_given=self.h_given,hrvc=self.hrvc)
            h[change_date] = pd.DataFrame(h_tmp,index=self.ret.columns)
        self.h = pd.Panel(h)
        self.h_rvc = pd.DataFrame(h_rvc,index=self.ret.columns).T
        self.h_es = pd.DataFrame(h_es,index=self.ret.columns).T
        self.others = pd.Panel(others)
        self.get_h_stat()
        if ifplot: self.plot_stack_weight()
        return self.h

    def get_h_stat(self):
        self.Ret = self.get_port_ret(ifcash=self.ifcash)[self.startdate:self.enddate]#各个method组合的收益率
        self.Ret_stat,self.Ret_nv,self.Ret_dd = Stat(ret=self.Ret,rf=self.rf).get_stat()
        #纳入交易成本计算
        cost = self.cost
        totalcost = {}
        for method in self.method_list:
            hh = self.h[:,:,method].T
            hhh = abs((hh-hh.shift(1)).dropna())
            q = self.Ret_nv[method][hh.index]
            totalcost[method] = (hh.apply(lambda x:x*q)*hhh*cost).sum().sum()
        totalcost = pd.Series(totalcost)
        annualcost = (1+totalcost)**(250/len(self.Ret_nv))-1
        new_stat = self.Ret_stat.copy()
        new_stat['annual_ret'] = self.Ret_stat['annual_ret'] - annualcost
        new_stat['sharpe'] = (new_stat['annual_ret']-self.Ret_stat['annual_ret']+new_stat['sharpe']*new_stat['annual_std'])/new_stat['annual_std']
        self.new_stat = new_stat
        return
    
        
    def get_change_data(self,change_date):
        #从df(dataframe)里找到change_date之前freq_len周期的数据，返回dataframe
        #df可以是价格数据，也可以是收益率数据
        startdate = self.change_list[self.change_list.index(change_date)-self.freq_len]
        return self.ret[startdate:change_date]
                            
    def __get_ret_leverage(self,l_list=['H11001.CSI','037.CS','000012.SH','000013.SH','H11008.CSI']):
        #加杠杆,l:杠杆率，cost:融资费率，是一个string 
        ret = self.ret.copy()
        for col in l_list:
            if col in ret.columns:
                ret[col] = ret[col]*self.leverage-self.rf*(self.leverage-1)
        return ret
        
    def plot_stack_weight(self,h=None):
        if h is None:
            try:
                h = self.h
            except:
                h = self.get_hh()
        if isinstance(h,pd.core.frame.DataFrame):
            h_tmp = {}
            h_tmp['weight'] = h
            h = pd.Panel(h_tmp).transpose(1,2,0)
        zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simkai.ttf')#自定义字体为楷体
        #画出权重的堆积曲线
    #    ax = hh.plot(kind='area')
    #    ax.set_ylim((0,1))
    #    ax.set_title(freq_dict[freq][0]+' change   datalength '+str(freq_len*freq_dict[freq][1])+'days')
        axes = {}
        for method in h.minor_axis:
            hh = h[:,:,method].T
            hh = hh.loc[:,hh.mean().sort_values(ascending=True).index]
            hh = hh.rename(columns=self.CODE_DICT)
            
            fig,ax = plt.subplots()
            #area图有个反人性的地方：legend的颜色顺序和图的颜色顺序相反，这里想办法设置一下
            a = ax.stackplot(hh.index,hh.T)
            try:
                ax.legend([i.decode('utf-8') for i in hh.columns[::-1]],prop=zhfont1)
            except:
                ax.legend(list(hh.columns[::-1]),prop=zhfont1)
            color = []
            for i in range(self.n):
                color.append(plt.getp(a[i],'facecolor'))
            color = color[::-1]
            for i in range(self.n):
                plt.setp(a[i],'facecolor',color[i])
            ax.set_xlim(tuple(hh.index[[0,-1]]))
            ax.set_ylim((0,1))
            plt.title(method)
            axes[method] = ax
        return h,axes
        
    def plot_net_value(self,net_value=None,linewidth=2):
        #画出净值曲线
        zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simkai.ttf')#自定义字体为楷体
        matplotlib.rcParams['lines.linewidth'] =linewidth
        if net_value is None:
            net_value = self.Ret_nv
        net_value = net_value.rename(columns=self.METHOD_DICT)
        net_value = net_value.rename(columns=self.CODE_DICT)
        net_value.plot()
        plt.legend(prop=zhfont1)
        return
        
    def plot_stat(self):
        def ret_stat(ret):
            result,_,_ = Stat(ret).get_stat()
            return result['total_ret']
        a = self
        result1 = a.Ret.groupby([a.Ret.index.year]).apply(ret_stat).rename(columns=a.METHOD_DICT)
        result2 = a.Ret.groupby([a.Ret.index.year,a.Ret.index.month]).apply(ret_stat).rename(columns=a.METHOD_DICT)
        X1 = result1.index
        X2 = np.arange(len(result2))
        for col in result1.columns:
            Y1 = result1.loc[:,col]
            width = 0.5
            f = plt.figure()
            ax1 = f.add_subplot(2,1,1)
            ax1.bar(X1,Y1,width,tick_label=X1,facecolor='lightskyblue',edgecolor='white',label=u'每年收益统计')
            plt.xticks(X1+width/2,X1)
            for x,y in zip(X1,Y1):
                plt.text(x+width/2, y, '%.2f%%' % (y*100), ha='center', va= 'bottom')
            ax1.legend()
            Y2 = result2.loc[:,col]
            ax2 = f.add_subplot(2,1,2)
            ax2.bar(X2,Y2,width,facecolor='lightskyblue',edgecolor='white',label=u'每月收益统计')
            ax2.legend()
        return
        
    def get_port_ret(self,ifcash=True):
    #返回按h的各个组合的收益率数据
    #ifcash表示是否以无风险利率投资于现金
        result = {}
        for method in self.method_list:
            ret = pd.Series()
            if type(method)==str:
                hh = self.h[:,:,method].T
            else:
                hh = self.h[:,:,getattr(method,'__name__')].T
            for change_date in hh.index[:-1]:
                #根据当前权重和下一个调仓周期的数据得到下一期净值走势
                #获取当前调仓日期下一日到下一个调仓日期的数据
                h_tmp = np.array(hh.ix[change_date])
                next_change_date = get_index_offset(hh,change_date,1)
                change_date = get_index_offset(self.ret,change_date,1)
                df_tmp = self.ret.ix[change_date:next_change_date].copy()
                if h_tmp.sum()<1:#以无风险利率投资于现金
                    try:
                        h_tmp = np.hstack((h_tmp,np.array([1-h_tmp.sum()])))
                        if ifcash:
                            df_tmp['cash'] = self.rf.loc[df_tmp.index]
                        else:
                            df_tmp['cash'] = 0
                    except:
                        pass
                x = ((df_tmp+1).cumprod()*h_tmp).sum(1).pct_change()*h_tmp.sum()
                x[0] = (df_tmp.iloc[0,:]*h_tmp).sum()
#                ret = ret.append((df_tmp*h_tmp).sum(1))
                ret = ret.append(x)
            #余下的数据的处理
            df_tmp = self.ret.ix[hh.index[-1]:].copy()
            h_tmp = np.array(hh.ix[-1])
            if h_tmp.sum()<1:
                try:
                    h_tmp = np.hstack((h_tmp,np.array([1-h_tmp.sum()])))
                    if ifcash:
                        df_tmp['cash'] = self.rf.loc[df_tmp.index]
                    else:
                        df_tmp['cash'] = 0
                except:
                    pass
            x = ((df_tmp+1).cumprod()*h_tmp).sum(1).pct_change()*h_tmp.sum()
            x[0] = (df_tmp.iloc[0,:]*h_tmp).sum()
            ret = ret.append(x[1:])
            if type(method)==str:
                result[method] = ret
            else:
                result[getattr(method,'__name__')] = ret
        result = pd.DataFrame(result)
        return result
        
        
class DB():
    def __init__(self,host = '127.0.0.1',
                      user = 'root',
                      password = '123123',
                      port = 3306,
                      database = 'wind',
                      charset = 'utf8'
                      ):
    
        self.host = host
        self.user = user
        self.password = password
        self.port = 3306
        self.database = database
        self.charset = 'utf8'

        self.conn = mysql.connector.connect(host = self.host,
                                            user = self.user,
                                            password = self.password,
                                            port = self.port,
                                            database = self.database,
                                            charset = self.charset)
        self.cursor=self.conn.cursor()
        self.cursor.execute("set interactive_timeout=24*3600")
        self.rows=None
        
    def select(self,string):
        self.cursor.execute(string)
        self.rows=self.cursor.fetchall()
        return self.rows

    def execute(self,string):
        self.cursor.execute(string)
        self.conn.commit()