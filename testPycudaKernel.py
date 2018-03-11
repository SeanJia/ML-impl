
import numpy as np
from numpy import *
import numpy.matlib as mat
import pycuda.autoinit
import pycuda.gpuarray as gpu
import pycuda.cumath as gpum
from pycuda import driver, compiler, tools

def createPrePostBackwardKernel(side_size, mini_batch_size, max_size):
    """create kernel for backprop part for pre post matrix"""

    kernel = """
    __global__ void PrePostBackwardKernel(float * AX, float * XB, float * D_AXB, float * D_pre, float * D_post,
                                           float * A, float * B, float * D_X) {

        // AX, XB are both sideSize*miniBatchSize by sideSize matrices,
        // D_AXB and D_X are both sideSize^2 by miniBatchSize matrices,
        // A, B are both sideSize by sideSize matrices.
        const uint sideSize = """ + str(side_size) + """;
        const uint miniBatchSize = """ + str(mini_batch_size) + """;
        const uint maxSize = """ + str(max_size) + """;
        const uint x = blockIdx.x * sideSize + threadIdx.x;        // blockSize in this case is by default sideSize
        if (x < maxSize) {
            __shared__ float rowInAD[sideSize];                    // row in A_T dot D_AXB
            uint sampleIdx = x / (sideSize * sideSize);
            uint temp = x % (sideSize * sideSize);
            uint rowIdx = temp / sideSize;
            uint colIdx = temp % sideSize;
            float sumPre = 0.0;
            float sumPost = 0.0;
            float sumAD = 0.0;
            float sumDX = 0.0;
            for (int i = 0; i < sideSize; i++) {
                sumPre += D_AXB[(rowIdx*sideSize + i)*miniBatchSize + sampleIdx] * XB[sampleIdx*sideSize*sideSize + colIdx*sideSize + i];
                sumPost += AX[sampleIdx*sideSize*sideSize + i*sideSize + rowIdx] * D_AXB[(colIdx + sideSize*i)*miniBatchSize + sampleIdx];
                sumAD += A[i*sideSize + rowIdx] * D_AXB[(colIdx + sideSize*i)*miniBatchSize + sampleIdx];
            }

            // set up the correct return values
            atomicAdd(&(D_pre[rowIdx * sideSize + colIdx]), sumPre);
            atomicAdd(&(D_post[rowIdx * sideSize + colIdx]), sumPost);

            // do (A_T dot D_AXB) dot B_T multiplication
            rowInAD[colIdx] = sumAD;
            __syncthreads();
            for (int i = 0; i < sideSize; i++) {
                sumDX += rowInAD[i] * B[colIdx * sideSize + i];
            }
            D_X[(rowIdx * sideSize + colIdx) * miniBatchSize + sampleIdx] = sumDX;
        }
    }
    """

    mod = compiler.SourceModule(kernel)
    return mod.get_function("PrePostBackwardKernel")

"""
D_AXB = np.random.randn(81, 100)
AX = np.random.randn(9*100, 9)
XB = np.random.randn(9*100, 9)
A = np.random.randn(9, 9)
B = np.random.randn(9, 9)
A_gpu = gpu.to_gpu(A.astype(np.float32))
B_gpu = gpu.to_gpu(B.astype(np.float32))
D_X = gpu.zeros((81, 100), np.float32)
AX_gpu = gpu.to_gpu(AX.astype(np.float32))
XB_gpu = gpu.to_gpu(XB.astype(np.float32))
D_AXB_gpu = gpu.to_gpu(D_AXB.astype(np.float32))
D_pre_gpu = gpu.zeros((9, 9), np.float32)
D_post_gpu = gpu.zeros((9, 9), np.float32)
D_pre_cpu = None
D_post_cpu = None
D_X_cpu = None
for i in range(100):
    curr_D_AXB = D_AXB[:, [i]]
    curr_XB = XB[9*i:9*(i+1), :]
    curr_res = np.reshape(curr_D_AXB, (9, 9)).dot(np.transpose(curr_XB))
    if i == 0:
        D_pre_cpu = curr_res
    else:
        #D_pre_cpu = np.concatenate((D_pre_cpu, curr_res), axis=0)
        D_pre_cpu += curr_res
for i in range(100):
    curr_D_AXB = D_AXB[:, [i]]
    curr_AX = AX[9*i:9*(i+1), :]
    curr_res = np.transpose(curr_AX).dot(np.reshape(curr_D_AXB, (9, 9)))
    if i == 0:
        D_post_cpu = curr_res
    else:
        #D_post_cpu = np.concatenate((D_post_cpu, curr_res), axis=0)
        D_post_cpu += curr_res
for i in range(100):
    curr_D_AXB = D_AXB[:, [i]]
    curr_res = np.reshape(np.transpose(A).dot(np.reshape(curr_D_AXB, (9, 9)).dot(np.transpose(B))), (81, 1))
    if i == 0:
        D_X_cpu = curr_res
    else:
        D_X_cpu = np.concatenate((D_X_cpu, curr_res), axis=1)
print D_X_cpu.shape
kernel = createPrePostBackwardKernel(9, 100, 9*9*100)
kernel(AX_gpu, XB_gpu, D_AXB_gpu, D_pre_gpu, D_post_gpu, A_gpu, B_gpu, D_X, grid=(900, 1, 1), block=(9, 1, 1))
print np.sum((D_X.get() - D_X_cpu) ** 2.0)
#print np.sum((D_post_gpu.get() - D_post_cpu) ** 2.0)
"""



def createPrePostForwardKernel(side_size, mini_batch_size, max_size):
    """create kernel for pre-post processing of input data in feed-forward pass"""

    kernel = """
    __global__ void ForwardKernel(float* A, float* B, float* X, float* AXB, float* AX, float* XB,
                                  int* pre_argmax, int* post_argmax) {

        // A for pre matrix, B for post matrix, X for data matrix, AXB is of side_size ^ 2 by mini_batch_size,
        // AX and XB are of side_size*mini_batch_size by side_size, and pre_argmax and post_argmax are integer arrays
        // of length sideSize. Notice that A and B should be matrices after a Rectified Linear Unit.
        const uint sideSize = """ + str(side_size) + """;
        const uint miniBatchSize = """ + str(mini_batch_size) + """;
        const uint maxSize = """ + str(max_size) + """;
        const uint x = blockIdx.x * sideSize + threadIdx.x;        // blockSize in this case is by default sideSize
        if (x < maxSize) {
            __shared__ float rowInAX[sideSize];
            uint sampleIdx = x / (sideSize * sideSize);
            uint temp = x % (sideSize * sideSize);
            uint rowIdx = temp / sideSize;
            uint colIdx = temp % sideSize;
            float sumAX = 0.0;
            float sumXB = 0.0;
            float sumAXB = 0.0;

            // for weights_pre (AX)
            if (pre_argmax[rowIdx] == 0) {
                float temp1 = A[rowIdx * sideSize];
                float temp2 = A[rowIdx * sideSize + 1];
                float rowSum = temp1 + temp2;
                if (rowSum != 0) {
                    sumAX = temp1 * X[colIdx*miniBatchSize+sampleIdx] +
                            temp2 * X[(colIdx+sideSize)*miniBatchSize+sampleIdx];
                    sumAX = sumAX / rowSum;
                }
            } else if (pre_argmax[rowIdx] == sideSize - 1) {
                float temp1 = A[rowIdx * sideSize + sideSize - 2];
                float temp2 = A[rowIdx * sideSize + sideSize - 1];
                float rowSum = temp1 + temp2;
                if (rowSum != 0) {
                    sumAX = temp1 * X[(colIdx+(sideSize-2)*sideSize)*miniBatchSize+sampleIdx] +
                            temp2 * X[(colIdx+(sideSize-1)*sideSize)*miniBatchSize+sampleIdx];
                    sumAX = sumAX / rowSum;
                }
            } else {
                int rowArgmax = pre_argmax[rowIdx];
                float temp1 = A[rowIdx * sideSize + rowArgmax - 1];
                float temp2 = A[rowIdx * sideSize + rowArgmax];
                float temp3 = A[rowIdx * sideSize + rowArgmax + 1];
                float rowSum = temp1 + temp2 + temp3;
                if (rowSum != 0) {
                    sumAX = temp1 * X[(colIdx+(rowArgmax-1)*sideSize)*miniBatchSize+sampleIdx] +
                            temp2 * X[(colIdx+rowArgmax*sideSize)*miniBatchSize+sampleIdx] +
                            temp3 * X[(colIdx+(rowArgmax+1)*sideSize)*miniBatchSize+sampleIdx];
                    sumAX = sumAX / rowSum;
                }
            }

            // prepare for AXB multiplication
            rowInAX[colIdx] = sumAX;
            __syncthreads();

            // for weights_post (XB)
            if (post_argmax[colIdx] == 0) {
                float temp1 = B[colIdx];
                float temp2 = B[sideSize + colIdx];
                float colSum = temp1 + temp2;
                if (colSum != 0) {
                    sumXB = X[rowIdx*sideSize*miniBatchSize+sampleIdx] * temp1 +
                            X[(rowIdx*sideSize+1)*miniBatchSize+sampleIdx] * temp2;
                    sumXB = sumXB / colSum;
                    sumAXB += rowInAX[0] * temp1 + rowInAX[1] * temp2;
                    sumAXB = sumAXB / colSum;
                }
            } else if (post_argmax[colIdx] == sideSize - 1) {
                float temp1 = B[(sideSize - 2) * sideSize + colIdx];
                float temp2 = B[(sideSize - 1) * sideSize + colIdx];
                float colSum = temp1 + temp2;
                if (colSum != 0) {
                    sumXB = X[(rowIdx*sideSize+sideSize-2)*miniBatchSize+sampleIdx] * temp1 +
                            X[(rowIdx*sideSize+sideSize-1)*miniBatchSize+sampleIdx] * temp2;
                    sumXB = sumXB / colSum;
                    sumAXB += rowInAX[sideSize-2] * temp1 + rowInAX[sideSize-1] * temp2;
                    sumAXB = sumAXB / colSum;
                }
            } else {
                int colArgmax = post_argmax[colIdx];
                float temp1 = B[(colArgmax - 1) * sideSize + colIdx];
                float temp2 = B[(colArgmax) * sideSize + colIdx];
                float temp3 = B[(colArgmax + 1) * sideSize + colIdx];
                float colSum = temp1 + temp2 + temp3;
                if (colSum != 0) {
                    sumXB = X[(rowIdx*sideSize+colArgmax-1)*miniBatchSize+sampleIdx] * temp1 +
                            X[(rowIdx*sideSize+colArgmax)*miniBatchSize+sampleIdx] * temp2 +
                            X[(rowIdx*sideSize+colArgmax+1)*miniBatchSize+sampleIdx] * temp3;
                    sumXB = sumXB / colSum;
                    sumAXB += rowInAX[colArgmax-1] * temp1 +
                              rowInAX[colArgmax] * temp2 +
                              rowInAX[colArgmax+1] * temp3;
                    sumAXB = sumAXB / colSum;
                }
            }

            // set up the correct return values
            AX[sampleIdx * sideSize * sideSize + rowIdx * sideSize + colIdx] = sumAX;
            XB[sampleIdx * sideSize * sideSize + rowIdx * sideSize + colIdx] = sumXB;
            AXB[(rowIdx * sideSize + colIdx) * miniBatchSize + sampleIdx] = sumAXB;
        }
    }
    """

    mod = compiler.SourceModule(kernel)
    return mod.get_function("ForwardKernel")

def forward_filter(w, sqrt_size, pre=True):
    result = np.zeros(w.shape)
    arr_argmax = []
    if pre:
        count = 0
        for row in w:
            argmax = np.argmax(row)
            arr_argmax.append(argmax)
            if argmax == 0:
                result[count, 0:2] = row[:2] / np.sum(row[:2])
            elif argmax == sqrt_size-1:
                result[count, -2:] = row[-2:] / np.sum(row[-2:])
            else:
                result[count, argmax-1:argmax+2] = row[argmax-1:argmax+2] / np.sum(row[argmax-1:argmax+2])
            count += 1
    else:
        count = 0
        w_t = w.T
        for col in w_t:
            argmax = np.argmax(col)
            arr_argmax.append(argmax)
            if argmax == 0:
                result[count, 0:2] = col[:2] / np.sum(col[:2])
            elif argmax == sqrt_size-1:
                result[count, -2:] = col[-2:] / np.sum(col[-2:])
            else:
                result[count, argmax - 1:argmax + 2] = col[argmax - 1:argmax + 2] / np.sum(col[argmax - 1:argmax + 2])
            count += 1
        result = result.T
    return result, arr_argmax


def ReLU(z):
    """the ReLU activation function"""
    return np.max([z, np.zeros(z.shape)], axis=0)


def relu_grad(z):
    """the gradient of ReLU(z)"""
    index = z >= 0
    result = np.zeros(z.shape)
    result[index] = 1.0
    return result


def createPrePostBackwardKernel(side_size, mini_batch_size, max_size):
    """create kernel for backprop part for pre post matrix"""

    kernel = """
    __global__ void BackwardKernel(float* AX, float* XB, float* D_AXB, float* temp_pre, float* temp_post, float* D_pre,
           float* D_post, float* A, float* B, float* D_X, int* pre_argmax, int* post_argmax, float* A_sp, float* B_sp) {

        // AX, XB are both sideSize*miniBatchSize by sideSize matrices,
        // D_AXB and D_X are both sideSize^2 by miniBatchSize matrices,
        // temp_pre/post are sideSize by sideSize matricies for middle step,
        // A, B are both sideSize by sideSize matrices AFTER ReLU;
        // the D_pre/post is the derivative after ReLU layer.
        // And A_sp/B_sp are w_pre/post after pooling layer.
        const uint sideSize = """ + str(side_size) + """;
        const uint miniBatchSize = """ + str(mini_batch_size) + """;
        const uint maxSize = """ + str(max_size) + """;
        const uint x = blockIdx.x * sideSize + threadIdx.x;        // blockSize in this case is by default sideSize
        if (x < maxSize) {
            __shared__ float rowInAD[sideSize];                    // row in A_T dot D_AXB
            uint sampleIdx = x / (sideSize * sideSize);
            uint temp = x % (sideSize * sideSize);
            uint rowIdx = temp / sideSize;
            uint colIdx = temp % sideSize;
            float sumPre = 0.0;
            float sumPost = 0.0;
            float sumAD = 0.0;
            float sumDX = 0.0;

            // obtain derivative through w_pre/post after the pooling filter
            for (int i = 0; i < sideSize; i++) {
                sumPre += D_AXB[(rowIdx*sideSize + i)*miniBatchSize + sampleIdx] * XB[sampleIdx*sideSize*sideSize + colIdx*sideSize + i];
                sumPost += AX[sampleIdx*sideSize*sideSize + i*sideSize + rowIdx] * D_AXB[(colIdx + sideSize*i)*miniBatchSize + sampleIdx];
                sumAD += A_sp[i*sideSize + rowIdx] * D_AXB[(colIdx + sideSize*i)*miniBatchSize + sampleIdx];
            }

            // merge the temp_pre/post for future use
            atomicAdd(&(temp_pre[rowIdx * sideSize + colIdx]), sumPre);
            atomicAdd(&(temp_post[rowIdx * sideSize + colIdx]), sumPost);

            // obtain D_pre from temp_pre; also obtain A_T dot D_AXB in the process
            if (pre_argmax[rowIdx] == 0) {
                if ((colIdx == 0) || (colIdx == 1)) {
                    float s = A[rowIdx*sideSize] + A[rowIdx*sideSize+1];
                    float ss = s * s;
                    if (ss != 0.0) {
                        if ((colIdx == 0) && (A[rowIdx*sideSize] > 0)) {
                            D_pre[rowIdx*sideSize] = A[rowIdx*sideSize+1] * (temp_pre[rowIdx*sideSize] - temp_pre[rowIdx*sideSize+1]) / ss;
                        } else if ((colIdx == 1) && (A[rowIdx*sideSize+1] > 0)) {
                            D_pre[rowIdx*sideSize+1] = A[rowIdx*sideSize] * (temp_pre[rowIdx*sideSize+1] - temp_pre[rowIdx*sideSize]) / ss;
                        }
                    }
                }
            } else if (pre_argmax[rowIdx] == sideSize - 1) {
                if ((colIdx == sideSize - 1) || (colIdx == sideSize - 2)) {
                    float s = A[rowIdx*sideSize+sideSize-1] + A[rowIdx*sideSize+sideSize-2];
                    float ss = s * s;
                    if (ss != 0.0) {
                        if ((colIdx == sideSize - 2) && (A[rowIdx*sideSize+sideSize-2] > 0)) {
                            D_pre[(rowIdx+1)*sideSize-2] = A[(rowIdx+1)*sideSize-1] * (temp_pre[(rowIdx+1)*sideSize-2] - temp_pre[(rowIdx+1)*sideSize-1]) / ss;
                        } else if ((colIdx == sideSize - 1) && (A[rowIdx*sideSize+sideSize-1] > 0)) {
                            D_pre[(rowIdx+1)*sideSize-1] = A[(rowIdx+1)*sideSize-2] * (temp_pre[(rowIdx+1)*sideSize-1] - temp_pre[(rowIdx+1)*sideSize-2]) / ss;
                        }
                    }
                }
            } else {
                int rowArgmax = pre_argmax[rowIdx];
                if ((colIdx == rowArgmax - 1) || (colIdx == rowArgmax) || (colIdx == rowArgmax + 1)) {
                    float s = A[rowIdx*sideSize+rowArgmax-1] + A[rowIdx*sideSize+rowArgmax] + A[rowIdx*sideSize+rowArgmax+1];
                    float ss = s * s;
                    if (ss != 0.0) {
                        float res = 0.0;
                        if ((colIdx == rowArgmax - 1) && (A[rowIdx*sideSize+rowArgmax-1] > 0)) {
                            res = (s - A[rowIdx*sideSize+rowArgmax-1]) * temp_pre[rowIdx*sideSize+rowArgmax-1] -
                                  A[rowIdx*sideSize+rowArgmax] * temp_pre[rowIdx*sideSize+rowArgmax] -
                                  A[rowIdx*sideSize+rowArgmax+1] * temp_pre[rowIdx*sideSize+rowArgmax+1];
                            D_pre[rowIdx*sideSize+rowArgmax-1] = res / ss;
                        } else if ((colIdx == rowArgmax) && (A[rowIdx*sideSize+rowArgmax] > 0)) {
                            res = -A[rowIdx*sideSize+rowArgmax-1] * temp_pre[rowIdx*sideSize+rowArgmax-1] +
                                  (s - A[rowIdx*sideSize+rowArgmax]) * temp_pre[rowIdx*sideSize+rowArgmax] -
                                  A[rowIdx*sideSize+rowArgmax+1] * temp_pre[rowIdx*sideSize+rowArgmax+1];
                            D_pre[rowIdx*sideSize+rowArgmax] = res / ss;
                        } else if ((colIdx == rowArgmax + 1) && (A[rowIdx*sideSize+rowArgmax+1] > 0)) {
                            res = -A[rowIdx*sideSize+rowArgmax-1] * temp_pre[rowIdx*sideSize+rowArgmax-1] -
                                  A[rowIdx*sideSize+rowArgmax] * temp_pre[rowIdx*sideSize+rowArgmax] +
                                  (s - A[rowIdx*sideSize+rowArgmax+1]) * temp_pre[rowIdx*sideSize+rowArgmax+1];
                            D_pre[rowIdx*sideSize+rowArgmax+1] = res / ss;
                        }
                    }
                }
            }

            // prepare for (A_T dot D_AXB) dot B_T multiplication
            rowInAD[colIdx] = sumAD;
            __syncthreads();

            // obtain D_post from temp_post; also obtain AD dot B_T in the process
            if (post_argmax[colIdx] == 0) {
                if ((rowIdx == 0) || (rowIdx == 1)) {
                    float s = B[colIdx] + B[sideSize+colIdx];
                    float ss = s * s;
                    if (ss != 0.0) {
                        if ((rowIdx == 0) && (B[colIdx] > 0)) {
                            D_post[colIdx] = B[sideSize+colIdx] * (temp_post[colIdx] - temp_post[sideSize+colIdx]) / ss;
                        } else if ((rowIdx == 1) && (B[sideSize+colIdx] > 0)) {
                            D_post[sideSize+colIdx] = B[colIdx] * (temp_post[sideSize+colIdx] - temp_post[colIdx]) / ss;
                        }
                    }
                }
            } else if (post_argmax[colIdx] == sideSize - 1) {
                if ((rowIdx == sideSize - 2) || (rowIdx == sideSize - 1)) {
                    float s = B[(sideSize-2)*sideSize + colIdx] + B[(sideSize-1)*sideSize + colIdx];
                    float ss = s * s;
                    if (ss != 0.0) {
                        if ((rowIdx == sideSize - 2) && (B[(sideSize-2)*sideSize+colIdx] > 0)) {
                            D_post[(sideSize-2)*sideSize+colIdx] = B[(sideSize-1)*sideSize+colIdx] *
                                (temp_post[(sideSize-2)*sideSize+colIdx] - temp_post[(sideSize-1)*sideSize+colIdx]) / ss;
                        } else if ((rowIdx == sideSize - 1) && (B[(sideSize-1)*sideSize+colIdx] > 0)) {
                            D_post[(sideSize-1)*sideSize+colIdx] = B[(sideSize-2)*sideSize+colIdx] *
                                (temp_post[(sideSize-1)*sideSize+colIdx] - temp_post[(sideSize-2)*sideSize+colIdx]) / ss;
                        }
                    }
                }
            } else {
                int colArgmax = post_argmax[colIdx];
                if ((rowIdx == colArgmax - 1) || (rowIdx == colArgmax) || (rowIdx == colArgmax + 1)) {
                    float s = B[(colArgmax-1)*sideSize+colIdx] + B[colArgmax*sideSize+colIdx] + B[(colArgmax+1)*sideSize+colIdx];
                    float ss = s * s;
                    if (ss != 0.0) {
                        float res = 0.0;
                        if ((rowIdx == colArgmax - 1) && (B[(colArgmax-1)*sideSize+colIdx] > 0)) {
                            res = (s - B[(colArgmax-1)*sideSize+colIdx]) * temp_post[(colArgmax-1)*sideSize+colIdx] -
                                  B[colArgmax*sideSize+colIdx] * temp_post[colArgmax*sideSize+colIdx] -
                                  B[(colArgmax+1)*sideSize+colIdx] * temp_post[(colArgmax+1)*sideSize+colIdx];
                            D_post[(colArgmax-1)*sideSize+colIdx] = res / ss;
                        } else if ((rowIdx == colArgmax) && (B[colArgmax*sideSize+colIdx] > 0)) {
                            res = -B[(colArgmax-1)*sideSize+colIdx] * temp_post[(colArgmax-1)*sideSize+colIdx] +
                                  (s - B[colArgmax*sideSize+colIdx]) * temp_post[colArgmax*sideSize+colIdx] -
                                  B[(colArgmax+1)*sideSize+colIdx] * temp_post[(colArgmax+1)*sideSize+colIdx];
                            D_post[colArgmax*sideSize+colIdx] = res / ss;
                        } else if ((rowIdx == colArgmax + 1) && (B[(colArgmax+1)*sideSize+colIdx] > 0)) {
                            res = -B[(colArgmax-1)*sideSize+colIdx] * temp_post[(colArgmax-1)*sideSize+colIdx] -
                                  B[colArgmax*sideSize+colIdx] * temp_post[colArgmax*sideSize+colIdx] +
                                  (s - B[(colArgmax+1)*sideSize+colIdx]) * temp_post[(colArgmax+1)*sideSize+colIdx];
                            D_post[(colArgmax+1)*sideSize+colIdx] = res / ss;
                        }
                    }
                }
            }

            // set up the correct return value for D_X
            for (int i = 0; i < sideSize; i++) {
                sumDX += rowInAD[i] * B_sp[colIdx * sideSize + i];
            }
            D_X[(rowIdx * sideSize + colIdx) * miniBatchSize + sampleIdx] = sumDX;
        }
    }
    """

    mod = compiler.SourceModule(kernel)
    return mod.get_function("BackwardKernel")


def backward(dw, pre=True, size=5, argmax=None, w=None):
    result = np.zeros(dw.shape)
    if pre:
        count = 0
        for rowInDw, rowInW in zip(dw, w):
            curr_argmax = argmax[count]
            if curr_argmax == 0:
                ss = (rowInW[0] + rowInW[1]) ** 2.0
                result[count, 0:2] = (1/ss) * np.array([rowInW[1]*(rowInDw[0]-rowInDw[1]), rowInW[0]*(rowInDw[1]-rowInDw[0])])
            elif curr_argmax == size-1:
                ss = (rowInW[-2] + rowInW[-1]) ** 2.0
                result[count, -2:] = (1/ss) * np.array([rowInW[-1]*(rowInDw[-2]-rowInDw[-1]), rowInW[-2]*(rowInDw[-1]-rowInDw[-2])])
            else:
                s = rowInW[curr_argmax-1] + rowInW[curr_argmax] + rowInW[curr_argmax+1]
                p1 = -rowInW[curr_argmax-1]*rowInDw[curr_argmax-1]
                p2 = -rowInW[curr_argmax]*rowInDw[curr_argmax]
                p3 = -rowInW[curr_argmax+1]*rowInDw[curr_argmax+1]
                result[count, curr_argmax-1:curr_argmax+2] = (1/(s ** 2.0)) * \
                      np.array([(s-rowInW[curr_argmax-1])*rowInDw[curr_argmax-1]+p2+p3,
                       p1+(s-rowInW[curr_argmax])*rowInDw[curr_argmax]+p3,
                       p1+p2+(s-rowInW[curr_argmax+1])*rowInDw[curr_argmax+1]])
            count += 1
    else:
        count = 0
        w_t = w.T
        dw_t = dw.T
        for colInDw, colInW in zip(dw_t, w_t):
            curr_argmax = argmax[count]
            if curr_argmax == 0:
                ss = (colInW[0] + colInW[1]) ** 2.0
                result[count, 0:2] = (1/ss) * np.array([colInW[1]*(colInDw[0]-colInDw[1]), colInW[0]*(colInDw[1]-colInDw[0])])
            elif curr_argmax == size-1:
                ss = (colInW[-2] + colInW[-1]) ** 2.0
                result[count, -2:] = (1/ss) * np.array([colInW[-1]*(colInDw[-2]-colInDw[-1]), colInW[-2]*(colInDw[-1]-colInDw[-2])])
            else:
                s = colInW[curr_argmax-1] + colInW[curr_argmax] + colInW[curr_argmax+1]
                p1 = -colInW[curr_argmax-1]*colInDw[curr_argmax-1]
                p2 = -colInW[curr_argmax]*colInDw[curr_argmax]
                p3 = -colInW[curr_argmax+1]*colInDw[curr_argmax+1]
                result[count, curr_argmax-1:curr_argmax+2] = (1/(s ** 2.0)) * \
                      np.array([(s-colInW[curr_argmax-1])*colInDw[curr_argmax-1]+p2+p3,
                       p1+(s-colInW[curr_argmax])*colInDw[curr_argmax]+p3,
                       p1+p2+(s-colInW[curr_argmax+1])*colInDw[curr_argmax+1]])
            count += 1
        result = result.T
    return result


# A_ori = ReLU(np.random.randn(9, 9))
# B_ori = ReLU(np.random.randn(9, 9))
# X = np.random.randn(81, 100)
# A_gpu = gpu.to_gpu(A_ori.astype(np.float32))
# B_gpu = gpu.to_gpu(B_ori.astype(np.float32))
# X_gpu = gpu.to_gpu(X.astype(np.float32))
# AX_gpu = gpu.zeros((900, 9), np.float32)
# XB_gpu = gpu.zeros((900, 9), np.float32)
# AXB_gpu = gpu.zeros((81, 100), np.float32)
# pre_argmax = np.argmax(A_ori, axis=1)
# post_argmax = np.argmax(B_ori, axis=0)
# pre_argmax_gpu = gpu.to_gpu(pre_argmax.astype(np.int32))
# post_argmax_gpu = gpu.to_gpu(post_argmax.astype(np.int32))
#
# A, pre_argmax_cpu = forward_filter(A_ori, 9, True)
# B, post_argmax_cpu = forward_filter(B_ori, 9, False)
# AX_cpu = None
# XB_cpu = None
# AXB_cpu = None
# for i in range(100):
#     curr_X = np.reshape(X[:, [i]], (9, 9))
#     AX = A.dot(curr_X)
#     if i == 0:
#         AX_cpu = AX
#     else:
#         AX_cpu = np.concatenate((AX_cpu, AX), axis=0)
#     XB = curr_X.dot(B)
#     if i == 0:
#         XB_cpu = XB
#     else:
#         XB_cpu = np.concatenate((XB_cpu, XB), axis=0)
#     AXB = np.reshape(AX.dot(B), (81, 1))
#     if i == 0:
#         AXB_cpu = AXB
#     else:
#         AXB_cpu = np.concatenate((AXB_cpu, AXB), axis=1)
#
# k = createPrePostForwardKernel(9, 100, 8100)
# k(A_gpu, B_gpu, X_gpu, AXB_gpu, AX_gpu, XB_gpu, pre_argmax_gpu, post_argmax_gpu, grid=(900, 1, 1), block=(9, 1, 1))
#
# print np.sum((AX_gpu.get() - AX_cpu) ** 2.0)
# print np.sum((XB_gpu.get() - XB_cpu) ** 2.0)
# print np.sum((AXB_gpu.get() - AXB_cpu) ** 2.0)


D_AXB = np.random.randn(81, 100)
AX = np.random.randn(9*100, 9)
XB = np.random.randn(9*100, 9)
A_ori = np.random.randn(9, 9)
B_ori = np.random.randn(9, 9)
A = ReLU(A_ori)
B = ReLU(B_ori)
A_gpu = gpu.to_gpu(A.astype(np.float32))
B_gpu = gpu.to_gpu(B.astype(np.float32))
D_X_gpu = gpu.zeros((81, 100), np.float32)
AX_gpu = gpu.to_gpu(AX.astype(np.float32))
XB_gpu = gpu.to_gpu(XB.astype(np.float32))
D_AXB_gpu = gpu.to_gpu(D_AXB.astype(np.float32))
D_pre_gpu = gpu.zeros((9, 9), np.float32)
D_post_gpu = gpu.zeros((9, 9), np.float32)
temp_pre_gpu = gpu.zeros((9, 9), np.float32)
temp_post_gpu = gpu.zeros((9, 9), np.float32)
pre_argmax = np.argmax(A, axis=1)
post_argmax = np.argmax(B, axis=0)
pre_argmax_gpu = gpu.to_gpu(pre_argmax.astype(np.int32))
post_argmax_gpu = gpu.to_gpu(post_argmax.astype(np.int32))

D_pre_cpu = None
D_post_cpu = None
D_X_cpu = None
for i in range(100):
    curr_D_AXB = D_AXB[:, [i]]
    curr_XB = XB[9*i:9*(i+1), :]
    curr_res = np.reshape(curr_D_AXB, (9, 9)).dot(np.transpose(curr_XB))
    curr_res = relu_grad(A_ori) * backward(curr_res, pre=True, size=9, argmax=np.argmax(A, axis=1), w=A)
    if i == 0:
        D_pre_cpu = curr_res
    else:
        D_pre_cpu += curr_res
for i in range(100):
    curr_D_AXB = D_AXB[:, [i]]
    curr_AX = AX[9*i:9*(i+1), :]
    curr_res = np.transpose(curr_AX).dot(np.reshape(curr_D_AXB, (9, 9)))
    curr_res = relu_grad(B_ori) * backward(curr_res, pre=False, size=9, argmax=np.argmax(B, axis=0), w=B)
    if i == 0:
        D_post_cpu = curr_res
    else:
        D_post_cpu += curr_res
for i in range(100):
    curr_D_AXB = D_AXB[:, [i]]
    A_, temp = forward_filter(A, 9, pre=True)
    B_, temp = forward_filter(B, 9, pre=False)
    curr_res = np.reshape((np.transpose(A_).dot(np.reshape(curr_D_AXB, (9, 9)))).dot(np.transpose(B_)), (81, 1))
    if i == 0:
        D_X_cpu = curr_res
    else:
        D_X_cpu = np.concatenate((D_X_cpu, curr_res), axis=1)
A_sp, temp = forward_filter(A, 9, pre=True)
B_sp, temp = forward_filter(B, 9, pre=False)
A_sp_gpu = gpu.to_gpu(A_sp.astype(np.float32))
B_sp_temp = np.zeros(B_sp.shape) + B_sp
B_sp_gpu = gpu.to_gpu(B_sp_temp.astype(np.float32))
for i in range(100):
    curr_D_AXB = D_AXB[:, [i]]
    curr_res = np.reshape(np.transpose(A_sp).dot(np.reshape(curr_D_AXB, (9, 9))), (81, 1))
    if i == 0:
        D_X_cpu_ = curr_res
    else:
        D_X_cpu_ = np.concatenate((D_X_cpu_, curr_res), axis=1)

kernel = createPrePostBackwardKernel(9, 100, 9*9*100)
kernel(AX_gpu, XB_gpu, D_AXB_gpu, temp_pre_gpu, temp_post_gpu, D_pre_gpu, D_post_gpu, A_gpu, B_gpu, D_X_gpu,
       pre_argmax_gpu, post_argmax_gpu, A_sp_gpu, B_sp_gpu, grid=(900, 1, 1), block=(9, 1, 1))

print np.sum((D_pre_cpu - D_pre_gpu.get()) ** 2.0)
print np.sum((D_post_cpu - D_post_gpu.get()) ** 2.0)
print np.sum((D_X_cpu - D_X_gpu.get()) ** 2.0)
