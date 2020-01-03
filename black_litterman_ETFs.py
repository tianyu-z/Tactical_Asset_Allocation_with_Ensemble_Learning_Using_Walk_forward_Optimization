import pandas
import numpy
numpy.random.seed(123)
import datetime
import yfinance as yf
import random
random.seed(2019)
import cvxopt as opt  
from cvxopt import blas, solvers  
solvers.options['show_progress'] = False


def bl_omega(conf, P, S):
    """
    This function computes the Black-Litterman parameters Omega from
       an Idzorek confidence.
    Inputs
      conf  - Idzorek confidence specified as a decimal (50% as 0.50)
      P     - Pick matrix for the view
      S     - Prior covariance matrix
    Outputs
      omega - Black-Litterman uncertainty/confidence parameter
    """

    alpha = (1 - conf) / conf
    omega = alpha * numpy.dot(numpy.dot(P, S), P.T)
    return omega


def create_view(view, pick_matrix, confidence, covariances, Q=None, P=None, O=None):
    """
    Creates a view, or updates views, for input to a Black-Litterman update
    
    Inputs
       view - the expected return of the trade
       pick_matrix - A pandas.Series identifying the trade to place
       confidence - A number in (0, 1) identifying our confidence in the trade
       covariance - The covariance matrix of assets in our universe
       Q - Vector of existing views
       P - Matrix of existing trades
       Q - Diagonal matrix of existing confidences
     
    Outputs
       Q - Updated vector of views
       P - Updated matrix of trades
       Q - Updated matrix of confidences
    """

    tickers = covariances.index
    n = len(tickers)

    assert(len(pick_matrix) <= n)

    # make sure all the assets in the pick matrix are in the covariance matrix
    assert(set(pick_matrix.index).issubset(set(tickers)))

    # are there items int he covariance matrix that are not in the pick matrix?
    if len(pick_matrix) < n:
        pick_matrix = pick_matrix.ix[tickers].fillna(0.)

    if type(pick_matrix) == type(pandas.Series):
        pick_matrix = pick_matrix.values

    if type(pick_matrix) != type(numpy.array):
        pick_matrix = numpy.array(pick_matrix)

    Q_is_None = type(Q) == type(None)
    P_is_None = type(P) == type(None)
    O_is_None = type(O) == type(None)

    if (P_is_None and Q_is_None and O_is_None):
        Q = pandas.Series(view, index = [0])

        P = pandas.DataFrame(pick_matrix.reshape(1, n), index = [0], columns = tickers)
        
        o = bl_omega(confidence, pick_matrix, covariances)
        
        O = pandas.DataFrame([o], index = [0], columns = [0])
        
    elif not (not P_is_None and not Q_is_None and not O_is_None):
        raise "P, Q, and O must all exist or be all None"

    else:

        k = len(Q)
        n = covariances.shape[0]

        assert(P.shape[0] == k)
        assert(P.shape[1] == n)

        assert(O.shape[0] == k)
        assert(O.shape[1] == k)

        Q = Q.append(pandas.Series([view], index = [k+1]))
        P = P.append(pandas.Series(pick_matrix, index = tickers), ignore_index = True)
    
        o = pandas.DataFrame([bl_omega(confidence, pick_matrix, covariances)], index = [k+1], columns = [k+1])
        O = O.append(o).fillna(0.)

    return Q, P, O
    

def black_litterman(pi, S, Q, P, O, t = 0.025):
    """
    Updates prior views with manager views
    
    Inputs:
       pi is our prior expected returns
       S is our prior covariance matrix
       Q is our vector of K views
       P is our matrix of trades associated with views (KxN)
       O is a diagonal matrix of uncertainty associated the the trades (KxK)
       t is the tau parameter (float)
    
    Outputs:
       Posterior Pi - Pi updated for views
       Posterior S - S updated for views
    """

    """assert(type(pi) == pandas.Series)
    assert(type(S) == pandas.DataFrame)
    assert(type(Q) == pandas.Series)
    assert(type(P) == pandas.DataFrame)
    assert(type(O) == pandas.DataFrame)"""

    tickers = pi.index

    k = len(Q)
    n = len(pi)

    assert(P.shape[0] == k)
    assert(P.shape[1] == n)

    assert(O.shape[0] == k)
    assert(O.shape[1] == k)

    pi = pi.values.T
    S = S.values
    Q = Q.values
    P = P.values
    O = O.values
    
    ts = t * S

    M = numpy.linalg.inv(ts) + numpy.dot(P.T, numpy.linalg.solve(O, P))

    posteriorPi = numpy.linalg.solve(M, (numpy.linalg.solve(ts, pi) + numpy.dot(P.T, numpy.linalg.solve(O, Q))))
    posteriorS = S + numpy.linalg.inv(M)

    posteriorPi = pandas.Series(posteriorPi.flatten(), index = tickers)
    posteriorS = pandas.DataFrame(posteriorS, index = tickers, columns = tickers)
   
    return [posteriorPi, posteriorS]


def optimal_portfolio(returns, covariance):
    n = len(returns)  
    returns = numpy.asmatrix(returns)  
    returns = returns.T
    N = 100  
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]  
    # Convert to cvxopt matrices  
    S = opt.matrix(covariance)  
    pbar = opt.matrix(returns)  
    # Create constraint matrices  
    G = -opt.matrix(numpy.eye(n))   # negative n x n identity matrix  
    h = opt.matrix(0.0, (n ,1))  
    A = opt.matrix(1.0, (1, n))  
    b = opt.matrix(1.0)  
    # Calculate efficient frontier weights using quadratic programming  
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x']  for mu in mus]  
    # CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]  
    risks = [numpy.sqrt(blas.dot(x, S*x)) for x in portfolios]  
    # CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = numpy.polyfit(returns, risks, 2)  
    x1 = numpy.sqrt(m1[2] / m1[0])  
    # CALCULATE THE OPTIMAL PORTFOLIO  
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']  
    return numpy.asarray(wt)


if __name__ == "__main__":

    tau = 0.025

    # old code# data = quantlab.pandas_ext.load_frames_into_frame_by_column(['shy', 'iei', 'ief'],
    # 'adjusted_close', 'data').resample('M').last()

    """
    # stocks = ['SHY', 'IEI', 'IEF','UDN','UUP','FXE','FXY','CEW','LQD','TLT','TIP','REM','EMB','EMLC','HYG','FLT','CMBS']

    start = datetime.datetime(2009, 8, 30)
    end = pandas.Timestamp.utcnow()
    http_proxy = "http://930257:Tuesday07%23@ncproxy1:8080/"
    data = yf.download(stocks, start=start, end=end, proxy=http_proxy)['Adj Close'].resample('M').last()
    expected_returns = data.apply(numpy.log).diff().mean() * 12.
    covariance = data.apply(numpy.log).diff().iloc[-60:].cov() * 12.

    # expected_returns.ix['shy'] = 0.0138
    # expected_returns.ix['iei'] = 0.0185
    # expected_returns.ix['ief'] = 0.0226

    # covariance.to_csv("cov.csv")
    """

    data = pandas.read_excel('Return Data.xlsx', header=[0, 1])
    data.columns = data.columns.get_level_values(0)
    data = data.rename(columns={'Unnamed: 0_level_0': 'Date'})
    data = data.sort_values(by='Date')
    data = data.drop(['Date'], axis=1)

    # data = data.set_index('Date')

    # print(data)
    # print(data.pct_change())

    # expected_returns = data.apply(numpy.log).diff().mean() * 12.
    expected_returns = data.pct_change().dropna().mean(axis=0) * 12
    covariance = data.pct_change().dropna().cov() * 12
    # covariance = data.apply(numpy.log).diff().iloc[-60:].cov() * 12.

    # expected_returns = expected_returns.sort_index()
    # covariance = covariance[expected_returns.index].loc[expected_returns.index]

    mu = expected_returns
    # print(expected_returns)
    # print(covariance)
    weights = []
    retour = 0

    # Here the analyst should put his view as input (his expectations),
    # since I don't have any expectation, I randomly generated it
    for i in range(expected_returns.shape[0]):
        a = random.uniform(-3, 3)
        weights.append(a)
        retour += a * mu.iloc[i]

    # print(retour)
    # print(weights)
    #(mu.ix['shy'] - mu.ix['iei'])

    tau = 0.025

    stocks = data.columns.values
    # print(stocks)

    # weights = [random.uniform(-3, 3) for _ in range(len(stocks))]
    # curve_trade = pandas.Series(weights, index=stocks).sort_index()
    curve_trade = pandas.Series(weights, index=stocks)

    # curve_trade = pandas.Series(weights)
    # print(len(weights))
    Q, P, O = create_view(-0.05, curve_trade, 0.5, covariance)

    print(Q.shape)
    print(P.shape)
    print(O.shape)

    # Here we use the Black-Litterman model to get the adjusted retruns and covariance
    posterior_er, posterior_covariance = black_litterman(expected_returns, covariance, Q, P, O, tau)

    print(posterior_er)
    print(posterior_covariance)

    mu_ = posterior_er
    retour_2 = 0

    for i in range(posterior_er.shape[0]):
        retour_2 += weights[i]*mu_.iloc[i]

    print(retour_2)
    # (mu_.ix['shy'] - mu_.ix['iei'])

    # Here we use the MVO to get the optimal weights
    op_weights = optimal_portfolio(mu_.values, posterior_covariance.values)
    print('optimal weights : \n', op_weights)
