import os
import sys
import ray
import h5py
import time
import math
import multiprocessing
import numpy as np
import scipy.stats
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import fdrcorrection as fdr
from statsmodels.regression import quantile_regression
from statsmodels.regression.quantile_regression import QuantReg
from sklearn import metrics
import numba as nb
from numba import jit, njit, prange
from copy import deepcopy
import shutil
try:
    import statsmodels.api as sm
except:
    import statsmodels as sm

##################################

def cp(file1, file2=None):
    """Copy a file to a new path.

    Parameters
    ----------
    file1 : str
        Source file path. If a single string with a space is provided it
        will be split into two paths for convenience.
    file2 : str or None
        Destination file path.
    """
    if file2 is None:
        ## check if a single string was given in, separated by a space
        temp_f_list = file1.split(" ")
        if len(temp_f_list)==2:
            file1 = temp_f_list[0]
            file2 = temp_f_list[1]
        else:
            print("\n\nsomething wrong with the cp syntax!\n\n")
            print("don't know how to interpret:",file1)
            sys.exit()
    with open(file1, 'rb') as f_in:
        with open(file2, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return()


def ray_get_indices_from_list(threads, original_idx_list):
    indices_list = []
    for t in range(threads):
        indices_list.append([])
    temp_idx = 0
    while temp_idx < len(original_idx_list):
        for t in range(threads):
            if temp_idx < len(original_idx_list):
                indices_list[t].append(original_idx_list[temp_idx])
                temp_idx += 1
    return(indices_list)


def get_bins(total_vars, bin_size=5000):
    bins = []
    cur_bin = 0
    while cur_bin<total_vars:
        bins.append(min(cur_bin, total_vars))
        cur_bin+=bin_size
    bins.append(total_vars)
    return(bins)


######################################################################
######################################################################
def pairwise_raw_pmd(in_mat):
    ###wrong o_actual = np.abs(in_mat[:,None,:]-in_mat[None,:,:])
    o_max = np.zeros((in_mat.shape[1],in_mat.shape[1]))
    np.fill_diagonal(o_max, np.sum(in_mat,axis=0))
######################################################################
######################################################################


@njit(fastmath=True)
def get_e(x):
    """Expected counts under independence for a contingency table.

    Parameters
    ----------
    x : ndarray of shape (n_rows, n_cols)
        Non-negative integer counts.

    Returns
    -------
    ndarray of shape (n_rows, n_cols)
        Expected counts matrix E = (row_sums / total) * col_sums.
    """
    row_sums = np.sum(x,axis=1)
    row_sums = row_sums/np.sum(row_sums)
    col_sums = np.sum(x,axis=0)
    n_rows =int(row_sums.shape[0])
    n_cols =int(col_sums.shape[0])
    out_e = np.zeros((n_rows,n_cols))
    for i in prange(n_rows):
        for j in range(n_cols):
            out_e[i,j] = row_sums[i]*col_sums[j]
    #out_e = np.dot(row_sums[:,None], col_sums[None,:])
    return(out_e)


## get_e(np.array([[100,0],[0,100]]))



@njit(fastmath=True)
def collate_vects(row_vect, col_vect, out_shape):
    out_mat = np.zeros(out_shape)
    for i in prange(row_vect.shape[0]):
        out_mat[row_vect[i],col_vect[i]]+=1
    return(out_mat)


def one_hot(in_vect):
    n_row = np.max(in_vect)
    n_col = in_vect.shape[0]
    out_mat = np.zeros((n_row+1,n_col),dtype=np.half)
    for i in range(out_mat.shape[0]):
        out_mat[i,:]=in_vect==i
    return(out_mat)


@njit(fastmath=True)
def get_null_vects(x, num_boot=1000):
    """Bootstrap PMD null by shuffling row labels.

    Parameters
    ----------
    x : ndarray
        Count matrix.
    num_boot : int, default=1000
        Number of bootstrap permutations.

    Returns
    -------
    ndarray of shape (num_boot,)
        Bootstrap samples of PMD values.
    """
    #x=np.array(x, x.dtype)#np.dtype(x.shape,))#,dtype=np.int32)
    all_counts = np.sum(x)
    row_vect = np.zeros((int(all_counts)), dtype=np.int32)
    col_vect = np.zeros((int(all_counts)), dtype=np.int32)
    ## get the base
    counter=0
    for i in prange(x.shape[0]):
        for j in range(x.shape[1]):
            for temp_count in range(x[i,j]):
                row_vect[counter] = i
                col_vect[counter] = j
                counter+=1
    # ## now shuffle and re-collate
    null_pmd = np.zeros((num_boot))#, dtype=numba.float32)
    # ##
    for temp_boot in range(num_boot):
        np.random.shuffle(row_vect)
        #temp_null_mat = collate_vects(r[:, np.random.permutation(r.shape[1])], c, x.shape)
        temp_null_mat = collate_vects(row_vect, col_vect, x.shape)######## to list?
        null_pmd[temp_boot] = np.float64(get_pmd(temp_null_mat))
    return(null_pmd)

#######################################################################
########### customized code to only do fast stdized resids ############
@njit(fastmath=True)
def get_null_diffs(x, expected, num_boot=100):
    #x=np.array(x, x.dtype)#np.dtype(x.shape,))#,dtype=np.int32)
    all_counts = np.sum(x)
    row_vect = np.zeros((int(all_counts)), dtype=np.int32)
    col_vect = np.zeros((int(all_counts)), dtype=np.int32)
    ## get the base
    counter=0
    for i in prange(x.shape[0]):
        for j in range(x.shape[1]):
            for temp_count in range(x[i,j]):
                row_vect[counter] = i
                col_vect[counter] = j
                counter+=1
    # ## now shuffle and re-collate
    # ##
    out_mat = np.zeros((x.shape[0],
                        x.shape[1],
                        nb.int64(num_boot)),
                       dtype=nb.float64)
    for temp_boot in range(num_boot):
        np.random.shuffle(row_vect)
        #temp_null_mat = collate_vects(r[:, np.random.permutation(r.shape[1])], c, x.shape)
        temp_mat = np.abs(collate_vects(row_vect, col_vect, x.shape) - expected)
        out_mat[:,:,temp_boot] = temp_mat.astype(np.float64)
    #null_mean = np.mean(out_mat, axis=2)
    #null_sd = np.std(out_mat, axis=2)
    return out_mat  # (null_mean, null_sd)



#@njit(fastmath=True)
def fast_std_resids(x, num_boot = 100, center_only=False):
    if type(x)==pd.DataFrame:
        x = x.to_numpy()
    expected = get_e(x)
    #null_mean, null_sd = get_null_diffs(x, expected, num_boot = num_boot)
    out_mat = get_null_diffs(x, expected, num_boot = num_boot)
    null_mean = np.mean(out_mat, axis=2)
    oe_actual = np.abs(x - expected)
    if center_only:
        return (oe_actual - null_mean)
    else:
        null_sd = np.std(out_mat, axis=2)
        temp_res = (oe_actual - null_mean)/null_sd
        temp_res[np.isnan(temp_res)]=0.
        temp_res[np.isinf(temp_res)]=0.
        return temp_res


#res = fast_std_resids(a, num_boot = 20)
#res_t = fast_std_resids(a.to_numpy().T, num_boot = 20)

#res.T - res_t
##################################################



#@njit(fastmath=True)#numba.cfunc(numba.int32[:,:],numba.uint32)
def get_detailed_null_vects(x, num_boot=1000):
    #x=np.array(x, x.dtype)#np.dtype(x.shape,))#,dtype=np.int32)
    all_counts = np.sum(x)
    row_vect = np.zeros((int(all_counts)), dtype=np.int32)
    col_vect = np.zeros((int(all_counts)), dtype=np.int32)
    ## get the base
    counter=0
    for i in prange(x.shape[0]):
        for j in range(x.shape[1]):
            for temp_count in range(x[i,j]):
                row_vect[counter] = i
                col_vect[counter] = j
                counter+=1
    # ## now shuffle and re-collate
    null_pmd = np.zeros((num_boot))#, dtype=numba.float32)
    null_residuals = np.zeros((num_boot, x.shape[0], x.shape[1]))
    # ##
    for temp_boot in range(num_boot):
        np.random.shuffle(row_vect)
        #temp_null_mat = collate_vects(r[:, np.random.permutation(r.shape[1])], c, x.shape)
        temp_null_mat = collate_vects(row_vect, col_vect, x.shape)######## to list?
        temp_pmd_res, temp_resid_mat = get_detailed_pmd(temp_null_mat)
        null_pmd[temp_boot] = np.float64(temp_pmd_res)
        null_residuals[temp_boot,:,:] = temp_resid_mat
    return(null_pmd, null_residuals)



@njit(fastmath=True)
def get_pmd(x):
    """Compute unadjusted PMD for a count matrix.

    Parameters
    ----------
    x : ndarray of shape (n_rows, n_cols)
        Non-negative integer counts.

    Returns
    -------
    float
        PMD ratio in [0, 1] where 1 indicates maximally different columns
        given the observed column marginals.
    """
    col_sums = np.sum(x,axis=0)
    if np.sum(col_sums)==0:
        print("sum(col_sums)==0!")
        print(x)
        return(np.nan)
    n_col = col_sums.shape[0]
    o_max = np.zeros((n_col,n_col),dtype=np.float32)
    for i in prange(n_col):
        o_max[i,i]=col_sums[i]
    #np.fill_diagonal(o_max,col_sums)
    #print(o_max)
    e_max = get_e(o_max)
    #print(e_max)
    max_delta = np.sum(np.abs(o_max-e_max))
    obs_e = get_e(x)
    obs_delta = np.sum(np.abs(x-obs_e))
    if max_delta==0:
        print("max_delta==1!")
        return(np.nan)
    ratio = obs_delta/max_delta
    return(ratio)


def get_detailed_pmd(x):
    """Return unadjusted PMD and residuals (observed - expected)."""
    temp_pmd = np.array(get_pmd(x),dtype=np.float64)
    temp_resid = np.array(x-get_e(x),dtype=np.float64)
    return(temp_pmd, temp_resid)

#get_pmd(np.array([[0,0],[0,0]]))
#get_pmd(np.array([[100,0],[0,100]]))


#get_null_vects(collate_vects(np.array([0,1,1,1,2,2,2]),np.array([0,1,1,1,2,2,1]),(3,3)))

@njit(fastmath=True)
def pmd_post_hoc_x(x, num_boot=1000, h5_out=None):
    """Pairwise unadjusted PMD between columns (Numba kernel)."""
    ##################################
    out_pmd = np.zeros((x.shape[1],x.shape[1]))
    for i in prange(x.shape[1]):
        for j in range(x.shape[1]):
            if i!=j:
                # get_pmd does not use num_boot; compute directly
                temp_res = np.float64(get_pmd(x[:,[i,j]]))
                out_pmd[i,j] = temp_res
                out_pmd[j,i] = temp_res
    return(out_pmd)



@jit(forceobj=True)
def pmd_post_hoc(pmd_res=None, x=None, num_boot=1000, h5_out=None):
    """Pairwise PMD between columns with optional HDF5 output.

    Parameters
    ----------
    pmd_res : pmd or None
        Existing result object; if provided, uses its `x`.
    x : ndarray or None
        Count matrix to compute pairwise PMD over.
    num_boot : int
        Number of bootstrap permutations used for debiasing.
    h5_out : str or None
        If provided, write results to this HDF5 path under dataset "infile".
    """
    ## returns either the path that was provided by the user to the output h5_out file, or
    ## if that was None, then it returns the actual pairwise PMD matrix
    if x is None and pmd_res is not None:
        x=pmd_res.x
    num_col=x.shape[1]
    ##################################
    if h5_out is not None:
        f=h5py.File(h5_out,'a')
        ## create the output dataset
        out_pmd = f.create_dataset("infile",
                                (num_col,num_col),
                                dtype=np.float32)
    else:
        out_pmd = np.zeros((num_col,num_col))
    ##################################
    for i in prange(num_col):
        for j in range(num_col):
            if i!=j:
                temp_res = pmd(x[:,[i,j]], num_boot=num_boot)
                out_pmd[i,j] = temp_res.pmd
                out_pmd[j,i] = temp_res.pmd
    if h5_out is not None:
        f.close()
        return(h5_out)
    else:
        return(out_pmd)


@ray.remote
def ray_pmd_post_hoc(in_h5, comparison_table, num_boot=25):
    #print("reading")
    f=h5py.File(in_h5,'r')
    shared_x=f["infile"]
    temp_comparisons = len(comparison_table)
    #out_pmd_dict = {}
    out_pmd_vect = np.zeros((temp_comparisons))
    ##################################
    #print("starting")
    for idx in range(temp_comparisons):
        #start = time.time()
        i = comparison_table[idx,0]
        j = comparison_table[idx,1]
        # try:
        #     dummy=out_pmd_dict[i]
        # except:
        #     out_pmd_dict[i]={}
        #temp_mat = np.concatenate((shared_x[:,i], shared_x[:,j]))
        temp_mat = shared_x[:,sorted([i,j])]
        #print(temp_mat)
        temp_res = pmd(temp_mat, num_boot=num_boot)
        #print(temp_res.pmd)
        #out_pmd_dict[i][j] = temp_res.pmd
        out_pmd_vect[idx] = temp_res.pmd
        #print(time.time()-start,"for run")
        if idx%10000==0:
            print("\t\t",100*idx/temp_comparisons,"%","in thread")
    f.close()
    #return(out_pmd_dict)
    return(out_pmd_vect)


@jit(forceobj=True)
def populate_out_mat_results(in_mat, ray_sults, temp_idxs):
    for thread in prange(len(ray_sults)):
        for idxs in range(temp_idxs[thread]):
            i=temp_idxs[thread][idxs,0]
            j=temp_idxs[thread][idxs,1]
            in_mat[i,j]=ray_sults[thread][idxs]
            in_mat[j,i]=ray_sults[thread][idxs]
    return(in_mat)


@njit
def collate_par_res(all_results, row_idxs, col_idxs):
    row_max = np.max(row_idxs)
    row_min = np.min(row_idxs)
    col_max = np.max(col_idxs)
    col_min = np.min(col_idxs)
    row_range = nb.int32(row_max-row_min)
    col_range = nb.int32(col_max-col_min)
    out_mat = np.zeros((row_range,col_range))
    for idx in prange(all_results.shape[0]):
        row_idx = row_idxs[idx]
        col_idx = col_idxs[idx]
        ## TODO: figure out if I'm going to return it offset with zeros 
        ## or just the smaller intact matrix
        out_mat[row_idx]
    return()


@njit(fastmath=True)
def get_pmd_pairs(a, b=None, num_boot=1000):
    """Pairwise debiased PMD between columns.

    If `b` is None, compute the symmetric pairwise PMD across columns of `a`.
    Otherwise compute the rectangular matrix comparing columns of `a` against `b`.

    Parameters
    ----------
    a : ndarray
        Count matrix A.
    b : ndarray or None
        Optional count matrix B. If omitted, B = A.
    num_boot : int
        Number of bootstrap permutations for null debiasing.

    Returns
    -------
    ndarray
        Pairwise PMD matrix.
    """
    ## goes through the columns of a, comparing them to column b
    # idxs in a are in rows, b will be in cols
    # if only a is supplied, it will give a symmetric a x a matrix
    #print("num_boot:",num_boot)
    print("doing post-hoc pairwise pmd comparison among columns")
    if b is None:
        b=a
        do_half = True
    else:
        do_half = False
    out_mat = np.zeros((a.shape[1], b.shape[1]))
    for row in prange(a.shape[1]):
        for col in range(b.shape[1]):
            if do_half:
                if row>col:
                    temp_mat = np.zeros((a.shape[0], 2))
                    temp_mat[:,0] = a[:,row]
                    temp_mat[:,1] = b[:,col]
                    #(self.raw_pmd-self.null_lambda)/(1-self.null_lambda)
                    null_lambda = np.mean( get_null_vects(temp_mat, num_boot = num_boot) ) 
                    if null_lambda==1:
                        print("null_lambda==1!")
                        out_mat[row, col] = np.nan
                    else:
                        out_mat[row, col] = (get_pmd(temp_mat)-null_lambda) / (1-null_lambda)
            else:
                temp_mat = np.zeros((a.shape[0], 2))
                temp_mat[:,0] = a[:,row]
                temp_mat[:,1] = b[:,col]
                #print(temp_mat)
                null_lambda = np.mean( get_null_vects(temp_mat, num_boot = num_boot) ) 
                if null_lambda==1:
                    print("null_lambda==1!")
                    out_mat[row, col] = np.nan
                else:
                    out_mat[row, col]= (get_pmd(temp_mat)-null_lambda) / (1-null_lambda)
    if do_half:
        #print(out_mat)
        ## dumb - but I don't know why += doesn't work here...
        out_mat = out_mat + out_mat.T
    return(out_mat)

#get_pmd_pairs(np.ones((100,10)), np.ones((100, 5)))

# start = time.time()
# ppair = get_pmd_pairs(np.random.poisson(5,100*100).reshape((100,100)), num_boot = 8)
# print(time.time()-start,"seconds")


@ray.remote
def ray_pmd_pairs(a, b=None, num_boot = 1000, idxs=None):
    return(get_pmd_pairs(a,b=b, num_boot = num_boot), idxs)


def get_sub_bins(num_sample_1, num_sample_2, threads):
    num_breaks = int(math.sqrt(threads))
    a_bins = num_breaks
    b_bins = num_breaks
    while a_bins*b_bins<threads:
        b_bins +=1
    a_breaks = get_bins(num_sample_1, int(num_sample_1/a_bins)+1)
    print("a_breaks",a_breaks)
    b_breaks = get_bins(num_sample_2, int(num_sample_2/b_bins)+1)
    print("b_breaks",b_breaks)
    a_start = a_breaks[:-1]
    a_end = a_breaks[1:]
    b_start = b_breaks[:-1]
    b_end = b_breaks[1:]
    print("sub_bins:")
    print(a_start)
    print(a_end)
    print(b_start)
    print(b_end)
    return(a_start, a_end, b_start, b_end)


def get_utril_bins(n):
    return(((n**2-n)/2)+n)


def get_num_bins(t):
    n = 0
    while get_utril_bins(n) < (t-1):
        n+=1
    if get_utril_bins(n)>t:
        n-=1
    return(n)


@jit(forceobj=True)
def get_parallel_ray_blocks(a, b=None, threads=None, num_boot = 1000):
    ## assumes ray.init() has already been called & will 
    ## be shutdown outside of this function
    ## a goes in rows, b goes in columns of the output
    if b is None:
        do_half = True
        b=a
    else:
        do_half = False 
    if do_half==True:
        num_bins =get_num_bins(threads)
        all_bins = get_bins(a.shape[1], int(a.shape[1]/num_bins)+1)
        print(all_bins)
        starts = all_bins[:-1]
        ends = all_bins[1:]
        #ray.init()
        ray_calls = []
        for i in range(len(starts)):
            for j in range(i,len(starts)):
                a_start = starts[i]
                a_end = ends[i]
                b_start = starts[j]
                b_end = ends[j]
                if i==j:
                    ray_calls.append(ray_pmd_pairs.remote(a[:,a_start:a_end], num_boot = num_boot, idxs=(a_start,a_end,b_start,b_end)))
                else:
                    ray_calls.append(ray_pmd_pairs.remote(a[:,a_start:a_end],b[:,b_start:b_end], num_boot = num_boot, idxs=(a_start,a_end,b_start,b_end)))
        ray_sults = ray.get(ray_calls)
        ## now log the results
        big_out_mat = np.zeros((a.shape[1],b.shape[1]))
        for element in ray_sults:
            a_start,a_end,b_start,b_end = element[1]
            if a_start == b_start and a_end == b_end:
                big_out_mat[a_start:a_end,a_start:a_end]=element[0]
            else:
                big_out_mat[a_start:a_end,b_start:b_end]=element[0]
                big_out_mat[b_start:b_end,a_start:a_end]=element[0].T
    else:
        #########################################
        a_start_list, a_end_list, b_start_list, b_end_list = get_sub_bins(a.shape[1], b.shape[1], threads)
        ray_calls=[]
        for i in range(len(a_start_list)):
            for j in range(len(b_start_list)):
                a_start = a_start_list[i]
                a_end = a_end_list[i]
                b_start = b_start_list[j]
                b_end = b_end_list[j]
                ray_calls.append(ray_pmd_pairs.remote(a[:,a_start:a_end],b[:,b_start:b_end], num_boot = num_boot, idxs=(a_start,a_end,b_start,b_end)))
        ray_sults = ray.get(ray_calls)
        big_out_mat = np.zeros((a.shape[1],b.shape[1]))
        for element in ray_sults:
            a_start,a_end,b_start,b_end = element[1]
            print("\nray call idxs:")
            print("a:",a_start, a_end)
            print("b:",b_start, b_end)
            print("fill shape:",element[0].shape, element[0].T.shape)
            print("target shape:",big_out_mat[a_start:a_end,b_start:b_end].shape)#, big_out_mat[b_start:b_end,a_start:a_end].T.shape)
            big_out_mat[a_start:a_end,b_start:b_end]=element[0]
    print(big_out_mat)
    return(big_out_mat)


# start = time.time()
# get_parallel_ray_blocks(np.random.poisson(5,100*1000).reshape((100,1000)),threads=10, num_boot=25)
# print("took",time.time()-start, "seconds")


def pad_output(out_pmd_mini_mat, 
               num_vars,
               b_start,
               b_end):
    ### here we create padded columns
    padded_mat = np.zeros((out_pmd_mini_mat.shape[0],num_vars))
    padded_mat[:,b_start:b_end] = out_pmd_mini_mat
    return(padded_mat)


@jit(forceobj=True)
def do_parallel_pmd_col_comparisons(in_mat, out_hdf5, block_size=5000, num_boot=8, threads=None, force=False):
    """Compute large pairwise PMD matrix in HDF5 blocks using Ray.

    Parameters
    ----------
    in_mat : str
        Path to an input HDF5 file with dataset "infile" (counts).
    out_hdf5 : str
        Path to output HDF5 file to write dataset "infile" (pairwise PMD).
    block_size : int
        Column block size for processing.
    num_boot : int
        Bootstrap iterations for debiasing within blocks.
    threads : int or None
        Number of Ray workers; defaults to CPU count.
    force : bool
        If False and output exists, skip computation.
    """
    #################################################
    if force==False and os.path.isfile(out_hdf5):
        print("\n"*10)
        print("found pmd output file & not forcing!")
        print("\n"*10)
        return()
    #################################################
    if threads == None:
        threads = multiprocessing.cpu_count()
    ###
    f_in=h5py.File(in_mat,'r')
    in_mat = f_in["infile"]
    num_col = in_mat.shape[1]
    block_size = min(block_size, num_col)
    max_num_comparisons_per_iter = ((block_size**2)-block_size)/2
    ####################################
    print("making the output: ",out_hdf5)
    f=h5py.File(out_hdf5,'a')
    ## create the output dataset
    out_pmd = f.create_dataset("infile",
                            (num_col,num_col),
                            dtype=np.half)
    ####################################
    total_comparisons = ((num_col**2)-num_col)/2
    num_rounds = math.floor(max_num_comparisons_per_iter/total_comparisons)+math.ceil(max_num_comparisons_per_iter%total_comparisons)
    #####################################################
    all_bins = get_bins(num_col, bin_size = block_size)
    starts = all_bins[:-1]
    ends = all_bins[1:]
    bin_count = 0
    total_num_bins = (((len(starts)**2)-len(starts))/2)+len(starts)
    ray.init()
    for i in range(len(starts)):
        for j in range(i,len(starts)):
            a_start = starts[i]
            a_end = ends[i]
            b_start = starts[j]
            b_end = ends[j]
            start_time = time.time()
            print("\t",round(100*bin_count/total_num_bins,2),"%","working on",a_start,"-",a_end, " vs ",b_start,"-",b_end)
            if i==j:
                temp_pmd_mat = get_parallel_ray_blocks(in_mat[:,a_start:a_end][:,:],
                                                       threads = threads,
                                                       num_boot = num_boot)
                end_time = time.time()
                print("\t\t\t",end_time-start_time,"computation")
                temp_pmd_mat = pad_output(temp_pmd_mat,
                                          out_pmd.shape[0],
                                          b_start,
                                          b_end)
                print("\t\t\t",time.time()-end_time,"padding")
                print("\t\tstoring")
                store_start = time.time()
                #out_pmd[a_start:a_end,:]= out_pmd[a_start:a_end,:] + temp_pmd_mat
                out_pmd[a_start:a_end,:] += temp_pmd_mat
                store_end = time.time()
                print("\t\t\t",store_end-store_start,"storing")
                ## don't add the transpose, because it's on the diagonal & will end
                ## up doubling the PMDs in this block!
                #out_pmd[a_start:a_end,:]= out_pmd[a_start:a_end,:] + temp_pmd_mat.T
            else:
                temp_pmd_mat = get_parallel_ray_blocks(in_mat[:,a_start:a_end][:,:],
                                                       b=in_mat[:,b_start:b_end][:,:],
                                                       threads = threads,
                                                       num_boot = num_boot)
                end_time = time.time()
                print("\t\t\t",end_time-start_time,"computation")
                temp_pmd_mat = pad_output(temp_pmd_mat,
                                          out_pmd.shape[0],
                                          b_start,
                                          b_end)
                print("\t\t\t",time.time()-end_time,"padding")
                store_start = time.time()
                #out_pmd[a_start:a_end,:] = out_pmd[a_start:a_end,:] + temp_pmd_mat
                #out_pmd[:,a_start:a_end] = out_pmd[:,a_start:a_end] + temp_pmd_mat.T
                out_pmd[a_start:a_end,:] += temp_pmd_mat
                out_pmd[:,a_start:a_end] += temp_pmd_mat.T
                #
                print("\t\t\t",time.time()-store_start,"storing")
            bin_count+=1
    ray.shutdown()
    #####################################################
    ####################################
    # ## make copies of the input hdf5 file
    # ray_copies=[]
    # print("making copies")
    # for t in range(threads):
    #     temp_file = in_mat+"_"+str(t)
    #     print("\t",temp_file)
    #     cp(in_mat,temp_file)
    #     ray_copies.append(temp_file)
    # #####################################
    # ray.init()
    # all_comparisons = []
    # round_counter=0
    # for i in range(num_col):
    #     for j in range(i,num_col):
    #         if i!=j:
    #             all_comparisons.append([i,j])
    #             if len(all_comparisons)==max_num_comparisons_per_iter:
    #                 print("\t",round(100*round_counter/num_rounds,2),"%")
    #                 round_counter+=1
    #                 ### do the analysis
    #                 temp_idxs_for_all_threads = ray_get_indices_from_list(threads,all_comparisons)
    #                 #print(temp_idxs)
    #                 ray_calls = []
    #                 for t in range(threads):
    #                     temp_idxs_for_all_threads[t] = np.array(temp_idxs_for_all_threads[t], dtype=int)
    #                     temp_idxs = temp_idxs_for_all_threads[t]
    #                     ray_calls.append(ray_pmd_post_hoc.remote(ray_copies[t], temp_idxs))
    #                 ray_sults = ray.get(ray_calls)
    #                 #out_pmd = populate_out_mat_results(out_pmd, ray_sults, temp_idxs)
    #                 print("\t\tstoring results")
    #                 for thread in range(len(ray_sults)):
    #                     temp_idxs = temp_idxs_for_all_threads[thread]
    #                     for idxs in range(temp_idxs.shape[0]):
    #                         i=temp_idxs[idxs,0]
    #                         j=temp_idxs[idxs,1]
    #                         out_pmd[i,j]=ray_sults[thread][idxs]
    #                 ray_sults=[]
    #                 gc.collect()
    #                 all_comparisons = []
    #                 round_counter+=1
    # ray.shutdown()
    # ###################################################
    f_in.close()
    f.close()
    return(out_hdf5)

####################################################################################################################
####################################################################################################################
def get_euc(in_pmd_h5, out_euc_h5, block_size = 5000, force = False):
    """Compute negative Euclidean distances over a PMD matrix in blocks.

    Parameters
    ----------
    in_pmd_h5 : str
        HDF5 file path containing pairwise PMD matrix under dataset "infile".
    out_euc_h5 : str
        Output HDF5 file path for negative Euclidean distances.
    block_size : int
        Block size for chunked computation.
    force : bool
        If False and output exists, reuse it.
    """
    euclidean_distances = metrics.pairwise.euclidean_distances
    in_pmd_f = h5py.File(in_pmd_h5,'a')
    in_pmd = in_pmd_f["infile"]
    total_vars=in_pmd.shape[0]
    print("total_vars",total_vars)
    bins = []
    cur_bin = 0
    while cur_bin<total_vars:
        bins.append(min(cur_bin, total_vars))
        cur_bin+=block_size
    bins.append(total_vars)
    print('making the negative euclidean distance matrix')
    ## make the euclidean distance output matrix
    ## make the hdf5 output file
    hdf5_euc_out_file = out_euc_h5
    print(hdf5_euc_out_file)
    euc_f = h5py.File(hdf5_euc_out_file, "a")
    ## set up the data matrix (this assumes float32)
    float_type = np.float16
    try:
        neg_euc_out_hdf5 = euc_f.create_dataset("infile", (total_vars,total_vars), dtype=float_type)
    except:
        neg_euc_out_hdf5 = euc_f["infile"]
    # else:
    #     neg_euc_out_hdf5 = euc_f.create_dataset("infile", (total_vars,total_vars), dtype=np.float32)
    # ## go through and calculate the negative euclidean distances
    ###########################
    spear_min = 99999
    temp_start=int(0)
    temp_end=int(block_size)
    print("calculating the minimum non-zero Spearman rho")
    while temp_end < in_pmd.shape[0]:
        temp_subset_mat = in_pmd[temp_start:temp_end,:]
        inf_idxs = np.isinf(temp_subset_mat)
        nan_idxs = np.isnan(temp_subset_mat)
        if np.sum(inf_idxs)+np.sum(nan_idxs)>0:
            temp_subset_mat[inf_idxs] = 0
            temp_subset_mat[nan_idxs] = 0
            in_pmd[temp_start:temp_end,:] = temp_subset_mat
        np.isinf(in_pmd[temp_start:temp_end,:])
        print(round(100*temp_start/in_pmd.shape[0],2),"% ","done")
        spear_min=min([spear_min,np.nanmin(in_pmd[temp_start:temp_end,:])/np.log2(total_vars)])
        temp_start+=block_size
        temp_end+=block_size
    ## catch the last bit
    spear_min=min([spear_min,np.nanmin(in_pmd[temp_start:,:])])
    #spear_min = np.nanmin(in_pmd)/np.log2(total_vars)
    print("spear_min",spear_min)
    num_compare = ((len(bins)**2) - len(bins))/2
    counter = -1
    ###########################
    for i in range(0,(len(bins)-1)):
        for j in range(i,(len(bins)-1)):
            #print(np.all(neg_euc_out_hdf5[bins[i]:bins[i+1],bins[j]:bins[j+1]]==0) , np.all(neg_euc_out_hdf5[bins[j]:bins[j+1],bins[i]:bins[i+1]]==0))
            counter+=1
            print(round(100*counter/num_compare,2),"% :",'euclidean distance for',bins[i],bins[i+1],'vs',bins[j],bins[j+1])
            num_zero_a = np.sum(np.abs(neg_euc_out_hdf5[bins[i]:bins[i+1],bins[j]:bins[j+1]]==0))
            num_zero_b = np.sum(np.abs(neg_euc_out_hdf5[bins[i]:bins[i+1],bins[j]:bins[j+1]]==0))
            if (num_zero_a+num_zero_b>2*block_size+100):
                #temp_neg_euc = -euclidean_distances(np.array(in_pmd[bins[i]:bins[i+1],:],dtype=np.float32),np.array(in_pmd[bins[j]:bins[j+1],:],dtype=np.float32),squared=True)
                #temp_neg_euc = -euclidean_distances(np.array(in_pmd[bins[i]:bins[i+1],:],dtype=float_type),np.array(in_pmd[bins[j]:bins[j+1],:],dtype=float_type),squared=True)/np.log2(total_vars)
                temp_subset_mat_1 = np.array(in_pmd[bins[i]:bins[i+1],:],dtype=float_type)/np.log2(total_vars)
                temp_subset_mat_2 = np.array(in_pmd[bins[j]:bins[j+1],:],dtype=float_type)/np.log2(total_vars)
                #print('before')
                #print(temp_subset_mat_1[np.where(np.isnan(temp_subset_mat_1))])
                temp_subset_mat_1[np.where(np.isnan(temp_subset_mat_1))]=spear_min
                #print(temp_subset_mat_1[np.where(np.isnan(temp_subset_mat_1))])
                #print('after')
                temp_subset_mat_2[np.where(np.isnan(temp_subset_mat_2))]=spear_min
                #print(temp_subset_mat_1)
                #print(temp_subset_mat_2)
                num_nan_1 = np.sum(np.isnan(temp_subset_mat_1))
                num_inf_1 = np.sum(np.isinf(temp_subset_mat_1))
                num_nan_2 = np.sum(np.isnan(temp_subset_mat_2))
                num_inf_2 = np.sum(np.isinf(temp_subset_mat_2))
                if num_nan_1+num_inf_1+num_nan_2+num_inf_2 >0:
                    print('num_nan_1:',num_nan_1)
                    print('num_inf_1:',num_inf_1)
                    print('num_nan_2:',num_nan_2)
                    print('num_inf_2:',num_inf_2)
                else:
                    pass#print('no nans or infs')
                temp_neg_euc = euclidean_distances(temp_subset_mat_1,temp_subset_mat_2)*np.log2(total_vars)
                neg_euc_out_hdf5[bins[i]:bins[i+1],bins[j]:bins[j+1]] = temp_neg_euc
                neg_euc_out_hdf5[bins[j]:bins[j+1],bins[i]:bins[i+1]] = np.transpose(temp_neg_euc)
                #print(temp_neg_euc)
                #print(neg_euc_out_hdf5[:,:])
                #neg_euc_out_hdf5 = np.transpose(temp_neg_euc)
                #print(np.all(neg_euc_out_hdf5[bins[i]:bins[i+1],bins[j]:bins[j+1]]==0) , np.all(neg_euc_out_hdf5[bins[j]:bins[j+1],bins[i]:bins[i+1]]==0))
            else:
                print('already finished',bins[i],bins[i+1],'vs',bins[j],bins[j+1])
    for i in range(0,np.shape(neg_euc_out_hdf5)[0]):
        neg_euc_out_hdf5[i,i]=-0.0
    in_pmd_f.close()
    euc_f.close()
    return(euc_f)


def z_mat_to_p_mat(z_mat):
    p_vals = scipy.stats.norm.sf(abs(z_mat))*2
    #print(p_vals)
    sig, corrected_p = fdr(p_vals.flatten())
    corrected_p=corrected_p.reshape(p_vals.shape)
    return(p_vals, corrected_p)


def cellwise_significance(resids, null_resids, mask_sd_cutoff=0.25):
    sd = np.std(null_resids,axis=0)
    ## the mean will always be zero centered (by definition)
    ## np.mean(null_resids,axis=0)
    ## so we can skip finding the distance to the mean for
    ## z-score calculation
    z_scores = resids/sd
    p_vals, corrected_p = z_mat_to_p_mat(z_scores)
    ## m[((m_pmd.residuals<0.25) & (m_pmd.z_scores<5))]
    ## here, we can get floating point errors below a certain sd as this in the 
    ## denominator will expload pretty quickly, creating artefactual significance
    ## just by having this error. We address this by simply masking with nans
    z_scores[sd<=mask_sd_cutoff]=np.nan
    return(z_scores, p_vals, corrected_p, sd)


# def adjust_one_col(resid_col, z_col, sd_col, row_sums, already_done_rows = [], max_adjust=.1):
#     if max_adjust is None:
#         max_adjust_percent = 1/np.sum(np.abs(resid_col))
#     ## handle the base case
#     if len(already_done_rows)==len(resid_col) or np.sum(row_sums[already_done_rows])/np.sum(row_sums)>max_adjust:
#         return(resid_col)
#     ## do the most significant row's adjustment that hasn't been done
#     current_row = [r for r in np.argsort(np.abs(z_col))[::-1] if r not in already_done_rows][0]
#     will_be_done_after_this_round = already_done_rows+[current_row]
#     remaining_rows = [r for r in range(len(resid_col)) if r not in will_be_done_after_this_round]
#     ## within that column, go through from most significant to 
#     ## least significant, calculating the number of "misplaced"
#     ## observations relative to the null & re-allocate them proportionally 
#     ## to the rest of the rows for that column
#     ## TODO -- left of here
#     #XXXtemp_non_row_idxs = np.array([r for r in range(x.shape[0]) if r!=row_idx])
#     #XXXfraction_remaining = np.sum(row_sums[temp_non_row_idxs])/total_sum
#     ## get the fraction of rows that aren't the current or previous & re-calculate the 
#     compositionally_displaced = resid_col[current_row]
#     ## relative remaining composition
#     remaining_sums = resid_col[remaining_rows]
#     remaining_proportions = remaining_sums/np.sum(remaining_sums)
#     ## now take the displaced cells & proportionally add them to all 
#     ## of the remaining rows
#     row_re_allocation = compositionally_displaced * remaining_proportions
#     print("other rows were:")
#     print(resid_col[remaining_rows])
#     resid_col[remaining_rows] += row_re_allocation
#     print("other rows are:")
#     print(resid_col[remaining_rows])
#     ###################
#     ## now re-calculate the z-scores for the whole thing
#     #z_col = resid_col / sd_col
#     return(
#         adjust_one_col(resid_col, z_col, sd_col, row_sums, already_done_rows = will_be_done_after_this_round)
#         )


def get_lm_adjust(z_col, quantile = .5):
    abs_res = np.abs(deepcopy(z_col))
    abs_res_sort_order = np.argsort(deepcopy(abs_res))
    cum_sum = np.zeros((abs_res_sort_order.shape))
    cum_sum[abs_res_sort_order] = np.cumsum(abs_res[abs_res_sort_order])
    ## remove top result
    subset_regression_include = sorted(abs_res_sort_order[:-1].tolist())
    endog = pd.DataFrame({"cum_sum":cum_sum[subset_regression_include]})
    exog = pd.DataFrame({"resid":z_col[subset_regression_include]})
    ## here we don't add the intercept, b/c it's supposed to be 0 intentionally
    dummy_order = np.argsort(endog["cum_sum"])
    #print(endog.iloc[dummy_order,:])
    #print(exog.iloc[dummy_order,:])
    #print(z_col)
    reg_fit = sm.OLS(endog, exog).fit()#, M=sm.robust.norms.HuberT()).fit()
    #print(reg_fit.params)
    regressed_resids = deepcopy(z_col) - (cum_sum*reg_fit.params[0])
    # plt.scatter(endog["cum_sum"],exog["resid"])
    # plt.scatter(cum_sum[subset_regression_include], z_col[subset_regression_include])
    # plt.plot([0,max(cum_sum)],[0,max(cum_sum)*reg_fit.params[0]])
    # plt.scatter(cum_sum, regressed_resids)
    # plt.show()
    return(deepcopy(regressed_resids))


# def robust_glm_correct(z_vect, quantile = 0.5):
#     if z_vect.shape[0]<3:
#         return(z_vect)
#     abs_z = np.abs(z_vect)
#     abs_z_order = np.argsort(abs_z)
#     abs_z_cum_sum = np.zeros(abs_z.shape)
#     abs_z_cum_sum[abs_z_order] = np.cumsum(abs_z[abs_z_order])
#     comb = pd.DataFrame(
#         {
#         "abs_z_cum_sum":abs_z_cum_sum,
#         "z_vect":z_vect
#         })
#     mod = smf.quantreg('z_vect ~ abs_z_cum_sum + 0', comb)
#     res = mod.fit(q=quantile)
#     slope = res.params["abs_z_cum_sum"]
#     regressed_res = z_vect-(abs_z_cum_sum*slope)
#     return(regressed_res)


def robust_glm_correct(z_vect, quantile=0.5):
    if z_vect.shape[0] < 3:
        return z_vect
    abs_z = np.abs(z_vect)
    abs_z_order = np.argsort(abs_z)
    abs_z_cum_sum = np.zeros(abs_z.shape)
    abs_z_cum_sum[abs_z_order] = np.cumsum(abs_z[abs_z_order])
    comb = pd.DataFrame(
        {
            "abs_z_cum_sum": abs_z_cum_sum,
            "z_vect": z_vect,
        }
    )
    mod = smf.quantreg("z_vect ~ abs_z_cum_sum + 0", comb)
    try:
        res = mod.fit(q=quantile)
        slope = res.params["abs_z_cum_sum"]
        regressed_res = z_vect - (abs_z_cum_sum * slope)
    except ValueError:
        # Add random Gaussian noise to z_vect and try again
        z_vect += np.random.normal(scale=1e-15, size=z_vect.shape)
        return robust_glm_correct(z_vect, quantile)
    return regressed_res


def get_compositional_resid(
    x, 
    resid_mat, 
    z_mat, 
    sd_mat, 
    previous_resids = None, 
    max_iters = 1, 
    relative_sse_epsilon = 0.01):
    previous_resids = deepcopy(resid_mat)
    adjusted_resid_mat = deepcopy(resid_mat)
    adjusted_z_mat = deepcopy(z_mat)
    ## calculate global row proportions
    row_sums = np.sum(x,axis=1)
    total_sum = np.sum(row_sums)
    row_proportions = row_sums/np.sum(row_sums)
    ## sort by greatest to least residuals
    sort_order = np.argsort(np.abs(z_mat),axis=0)
    for j in prange(x.shape[1]):
        ## go through the columns & do the adjustment
        if False:
            for row_idx in sort_order[:,j]:#[::-1]:
                ## within that column, go through from most significant to 
                ## least significant, calculating the number of "misplaced"
                ## observations relative to the null & re-allocate them proportionally 
                ## to the rest of the rows for that column
                temp_non_row_idxs = np.array([r for r in range(x.shape[0]) if r!=row_idx])
                fraction_remaining = np.sum(row_sums[temp_non_row_idxs])/total_sum
                compositionally_displaced = adjusted_resid_mat[row_idx,j]
                print("sd_mat[row_idx,j]:",sd_mat[row_idx,j])
                ## calculate the percent error
                sd_displaced = adjusted_resid_mat[row_idx,j] / (sd_mat[row_idx,j])
                print("\n\nrow_idx:",row_idx,"col_idx:",j,"\tdisplaced:",compositionally_displaced)
                ## get the fraction of rows that aren't the current & re-calculate the 
                ## relative remaining composition
                remaining_sums = row_sums[temp_non_row_idxs]
                remaining_proportions = remaining_sums/np.sum(remaining_sums)
                ## now take the displaced cells & proportionally add them to all 
                ## of the remaining rows
                row_re_allocation = compositionally_displaced * remaining_proportions
                print("other rows were:")
                print(adjusted_resid_mat[temp_non_row_idxs,j])
                adjusted_resid_mat[temp_non_row_idxs,j] += row_re_allocation
                print("other rows are:")
                print(adjusted_resid_mat[temp_non_row_idxs,j])
        else:
            #adjusted_resid_mat[:,j] = adjust_one_col(adjusted_resid_mat[:,j], z_mat[:,j], sd_mat[:,j], row_sums, already_done_rows = [])
            #adjusted_z_mat[:,j] = get_lm_adjust(deepcopy(adjusted_z_mat[:,j]))
            adjusted_z_mat[:,j] = robust_glm_correct(deepcopy(adjusted_z_mat[:,j]).flatten())
            ####
    #adjusted_z_mat = adjusted_resid_mat / sd_mat
    adjusted_p, adjusted_corrected_p = z_mat_to_p_mat(adjusted_z_mat)
    #relative_sse = np.sum(((previous_resids-adjusted_resid_mat)**2)/np.sum(adjusted_resid_mat**2))
    #print("iter:",max_iters,"sse:",relative_sse)
    adjusted_residuals = adjusted_z_mat * sd_mat
    return(adjusted_resid_mat, adjusted_z_mat, adjusted_p, adjusted_corrected_p)
    # if relative_sse < relative_sse_epsilon or max_iters == 0:
    #     return(adjusted_resid_mat, adjusted_z_mat, adjusted_p, adjusted_corrected_p)
    # else:
    #     return(get_compositional_resid(x, 
    #                                     deepcopy(adjusted_resid_mat), 
    #                                     deepcopy(adjusted_z_mat), 
    #                                     sd_mat, 
    #                                     max_iters = max_iters-1, 
    #                                     relative_sse_epsilon = relative_sse_epsilon)
    #     )




# adjusted_resid_mat, adjusted_z_mat, adjusted_p, adjusted_corrected_p = get_compositional_resid(
#     pbmc_pmd.x.to_numpy(), 
#     pbmc_pmd.residuals.to_numpy(), 
#     pbmc_pmd.z_scores.to_numpy(), 
#     pbmc_pmd.residual_sd.to_numpy())
# adjusted_z_df = pd.DataFrame(adjusted_z_mat, index = pbmc_pmd.residuals.index, columns = pbmc_pmd.residuals.columns)



##cellwise_significance(pmd_res.residuals, pmd_res.null_residuals)


####################################################################################################################
####################################################################################################################

class pmd(object):
    def __init__(self, x, y=None, num_boot=1000, skip_posthoc=False):
        """Compute PMD and related statistics for a count matrix.

        Parameters
        ----------
        x : numpy.ndarray or pandas.DataFrame
            Non-negative integer counts with shape (n_features, n_samples).
            If a DataFrame is provided, row/column labels are preserved.
        y : unused
            Reserved for future use.
        num_boot : int, default 1000
            Number of bootstrap permutations to estimate the null.
        skip_posthoc : bool, default False
            If True, skip pairwise post-hoc PMD between columns to save time.

        Attributes
        ----------
        pmd : float
            Debiased PMD in approximately [0, 1].
        raw_pmd : float
            Unadjusted PMD before null debiasing.
        null : ndarray
            Bootstrap samples of PMD under the null.
        null_lambda : float
            Mean of the null distribution.
        p_val : float
            One-sided p-value relative to the null.
        residuals : pandas.DataFrame
            Observed minus expected counts under independence.
        z_scores, p_vals, corrected_p_vals : pandas.DataFrame
            Cell-wise significance measures.
        post_hoc : pandas.DataFrame
            Pairwise PMD between columns (if computed).
        """
        # Basic validation and coercion
        if "index" in dir(x):
            rows = x.index
            cols = x.columns
        else:
            rows = np.arange(x.shape[0],dtype=np.int32)
            cols = np.arange(x.shape[1],dtype=np.int32)
        self.x = np.array(x,dtype=np.int64)
        if self.x.ndim != 2:
            raise ValueError("x must be a 2D count matrix")
        if np.any(self.x < 0):
            raise ValueError("x must contain non-negative counts")
        ## if x is a 2d matrix
        ## if x and y are count vectors
        #self.raw_pmd = get_pmd(x)
        self.raw_pmd, self.residuals = get_detailed_pmd(self.x)
        #self.null = get_null_vects(x, num_boot=num_boot)
        self.null, self.null_residuals = get_detailed_null_vects(self.x, num_boot=num_boot)
        self.null_lambda = np.mean(self.null)
        self.pmd = (self.raw_pmd-self.null_lambda)/(1-self.null_lambda)
        if self.pmd > 0:
            self.p_val = np.sum(self.raw_pmd > self.null_lambda)/num_boot
        else:
            self.p_val = np.sum(self.raw_pmd < self.null_lambda)/num_boot
        self.z_scores, self.p_vals, self.corrected_p_vals, self.residual_sd = cellwise_significance(self.residuals, self.null_residuals)
        if not skip_posthoc:
            self.post_hoc = get_pmd_pairs(self.x, num_boot = num_boot)
        else:
            self.post_hoc = np.zeros((len(cols),len(cols)))
        ## now do the compositional adjustment
        ## Tried to address the compositional problem, but this specific approach didn't work
        # It turned out to be perfect when there was only *1* group that was different, but
        # as soon as there were 2, the compositional problem was back...
        # There could be a solution hiding in here deeper, but I haven't found it yet...
        adjusted_resid_mat, adjusted_z_mat, adjusted_p, adjusted_corrected_p = get_compositional_resid(
            self.x, 
            self.residuals, 
            self.z_scores, 
            self.residual_sd
        )
        ## convert to pandas
        self.z_scores = pd.DataFrame(self.z_scores, index = rows, columns = cols)
        self.p_vals = pd.DataFrame(self.p_vals, index = rows, columns = cols)
        self.corrected_p_vals = pd.DataFrame(self.corrected_p_vals, index = rows, columns = cols)
        self.residuals = pd.DataFrame(self.residuals, index = rows, columns = cols)
        self.residual_sd = pd.DataFrame(self.residual_sd, index = rows, columns = cols)
        self.post_hoc = pd.DataFrame(self.post_hoc, index = cols, columns = cols)
        self.x = pd.DataFrame(self.x, index = rows, columns = cols)
        self.adjusted_residuals = pd.DataFrame(adjusted_resid_mat, index = rows, columns = cols)
        self.adjusted_z_scores  = pd.DataFrame(adjusted_z_mat, index = rows, columns = cols)
        self.adjusted_p_vals = pd.DataFrame(adjusted_p, index = rows, columns = cols)
        self.adjusted_corrected_p_vals = pd.DataFrame(adjusted_corrected_p, index = rows, columns = cols)



def do_demo():
    n_col = 500
    n_row = 25
    a=np.zeros((n_row,n_col),dtype=np.int64)
    for i in range(n_col):
        a[:,i]=np.random.poisson(lam=max(0,np.random.normal(5,10,1))*((1+np.random.uniform())**4),size=(n_row))
        a=pd.DataFrame(a,
            index = ["row"+str(i) for i in range(n_row)],
            columns = ["col"+str(i) for i in range(n_col)],)
        pmd_res = pmd(a)
        return(pmd_res)


#r = do_demo()
