import os, glob, re, signal, sys, argparse, threading, time
os.environ["GLOG_minloglevel"] = "1"
import numpy as np
import scipy.misc as sc
import sys
sys.path.append('/data2/caffe/python') #path of pycaffe
import caffe
import math
def ensure_dir(f):
    try:
        os.makedirs(f)
    except OSError:
        pass
#weights = '/data3/FreeWork/caffe-vdsr/Train/VDSR_Adam.caffemodel'
weights = './Models/VDSR_Official.caffemodel'
model = './Models/VDSR_net_deploy.prototxt'

parser = argparse.ArgumentParser()

parser.add_argument("--im_list", help="Path for input file")
parser.add_argument("--out_path", help="Dir for output file")
parser.add_argument("--scale", help="Upscale Factor")
parser.add_argument("--batch_size", help="size of batch")
args = parser.parse_args()
from skimage import io, exposure, img_as_uint, img_as_float
io.use_plugin('freeimage')
im_path_file = args.im_list
out_path = args.out_path
if not out_path: 
	out_path = os.path.dirname(im_path)
else:
	ensure_dir(out_path)
scale = int(args.scale)
batch_size = int(args.batch_size)
#im_path = '/data3/FreeWork/caffeVDSR/Test/Set14/flowers.bmp'
#out_path = './Results/'
#scale = 2


gpu_mode = 1

def rgb2ycbcr(im):
    xform = np.array([[65.481/255.0, 128.553/255.0, 24.966/255.0], [-37.797/255.0, -74.203/255.0, 112/255.0], [112/255.0, -93.786/255.0, -18.214/255.0]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,0] += 16.0/255.0
    ycbcr[:,:,[1,2]] += 128.0/255.0
    return np.double(ycbcr)

from skimage import measure
def compute_psnr(img1,img2):
	'''
	if len(img1.shape)==3:
		img1 = rgb2ycbcr(img1)
		img1 = img1[:,:,0]
	if len(img2.shape)==3:
		img2 = rgb2ycbcr(img2)
		img2 = img2[:,:,0]
	imdff = 
	'''
	psnr = measure.compare_psnr(img1,img2)
	ssim = measure.compare_ssim(img1,img2,multichannel=True)
	return psnr,ssim

def modcrop(img, modulo):
	tmpsz = img.shape
	sz = tmpsz[0:2]
	sz = sz - np.array([sz[0]%modulo,sz[1]%modulo])	
	return img[0:sz[0],0:sz[1],:]

def ycbcr2rgb(im):
    #xform = np.array([[1.164, 0, 1.596], [1.164, -0.392, -.813], [1.164, 2.017, 0]])
    xform = np.load('ycbcr.npy')
    rgb = im.astype(np.float)
    rgb[:,:,0] -= 16.0/255.0
    rgb[:,:,[1,2]] -= 128.0/255.0
    return np.double(rgb.dot(xform.T))

def generateInputs(inputs):
	size = [256,256]
	
	if inputs.shape[0] < size[0]:
		size[0] = inputs.shape[0]
	if inputs.shape[1] < size[1]:
		size[1] = inputs.shape[1]
	hPad, wPad = inputs.shape[0] % size[0],inputs.shape[1] % size[1]
	hAdd, wAdd = 0, 0
	if hPad != 0:
		hAdd = (size[0] - hPad)
	if wPad !=0:
		wAdd = (size[1] - wPad)
	quantity = ((inputs.shape[0]+hAdd)/size[0]) * ((inputs.shape[1] +wAdd)/size[1])
	data_full = np.empty((quantity,1,size[0],size[1]))
	shape1 = inputs.shape[0]+hAdd
	shape2 = inputs.shape[1]+wAdd
	count = 0
	for x in range(0,shape1-size[0]+1, size[0]):
		for y in range(0,shape2-size[1]+1,size[1]):	
			if (x+size[0]) <= inputs.shape[0] and (y+size[1]) <= inputs.shape[1]:
				data_full[count,0] = inputs[x:x+size[0],y:y+size[1]]
			elif (x+size[0]) <= inputs.shape[0] and (y+size[1]) > inputs.shape[1]:
				data_full[count,0] = inputs[x:x+size[0],inputs.shape[1]-size[1]:]
			elif (x+size[0]) > inputs.shape[0] and (y+size[1]) <= inputs.shape[1]:
				data_full[count,0] = inputs[(inputs.shape[0]-size[0]):,y:y+size[1]]
			else:
				data_full[count,0] = inputs[(inputs.shape[0]-size[0]):,(inputs.shape[1]-size[1]):]
			count = count + 1
	assert(count==quantity)
	return data_full

def generateOutput(outputs, size_original):

	size = [outputs.shape[2],outputs.shape[3]]
	hPad, wPad = size_original[0] % size[0],size_original[1] % size[1]
	hAdd, wAdd = 0, 0
	if hPad != 0:
		hAdd = (size[0] - hPad)
	if wPad !=0:
		wAdd = (size[1] - wPad)

	quantity = outputs.shape[0]
	data_label = np.empty((size_original[0],size_original[1]))
	count =0
	shape1 = size_original[0]+hAdd
	shape2 = size_original[1]+wAdd
	for x in range(0,shape1-size[0]+1, size[0]):
		for y in range(0,shape2-size[1]+1,size[1]):	
			if (x+size[0]) <= size_original[0] and (y+size[1]) <= size_original[1]:
				data_label[x:x+size[0],y:y+size[1]] = outputs[count,0]
			elif (x+size[0]) <= size_original[0] and (y+size[1]) > size_original[1]:
				data_label[x:x+size[0],y:] = outputs[count,0,:,y+size[1]-size_original[1]:]
			elif (x+size[0]) > size_original[0] and (y+size[1]) <= size_original[1]:
				data_label[x:,y:y+size[1]] = outputs[count,0,x+size[0]-size_original[0]:,:]
			else:
				data_label[x:,y:] = outputs[count,0,x+size[0]-size_original[0]:,y+size[1]-size_original[1]:]
			count = count + 1
	assert(count==quantity)
	return data_label


def patch_VDSR(net, input_in, batch):
    
    net_in = generateInputs(input_in)

    for i in range(int(math.ceil(1.0*net_in.shape[0]/batch))):
	start=i*batch
	end = min(net_in.shape[0], start+batch)
    	net.blobs['data'].reshape(*net_in[start:end].shape)
    	net.blobs['data'].data[...] = net_in[start:end]
    	net.forward()
    	net_in[start:end] =  np.array(net.blobs['sum'].data[:,:,:,:])
    
    return generateOutput(net_in, input_in.shape)
    

def process(im_path, net, scale,batch):
    file=im_path
    orig_gt = sc.imread(file)
    if len(orig_gt.shape) == 3:
        mode='RGB'
    else:
        mode='L'

    #orig_gt = modcrop(orig_gt, scale)

    
    
    orig_hi = orig_gt
    #print(compute_psnr(orig_gt,orig_hi))
    #Get Y channel of Interpolated Image
    orig_hi = orig_hi.astype(np.float32)/255.0
    if mode=='RGB':
    	orig_hi_ycbcr_1 = rgb2ycbcr(orig_hi)
	orig_hi_ycbcr = np.empty((orig_gt.shape[0] * scale, orig_gt.shape[1] * scale, 3),dtype=np.float32)
	orig_hi_ycbcr[:,:,0] = sc.imresize(orig_hi_ycbcr_1[:,:,0], (orig_gt.shape[0] * scale, orig_gt.shape[1] * scale),
                                       interp='bicubic', mode='F')
        orig_hi_ycbcr[:, :, 1] = sc.imresize(orig_hi_ycbcr_1[:, :, 1], (orig_gt.shape[0] * scale, orig_gt.shape[1] * scale),
                                         interp='bicubic', mode='F')
        orig_hi_ycbcr[:, :, 2] = sc.imresize(orig_hi_ycbcr_1[:, :, 2], (orig_gt.shape[0] * scale, orig_gt.shape[1] * scale),
                                         interp='bicubic', mode='F')

        orig_hi_y = orig_hi_ycbcr[:,:,0]
    else:
    	orig_hi_ycbcr_1 = orig_hi
	orig_hi_ycbcr = np.empty((orig_gt.shape[0] * scale, orig_gt.shape[1] * scale),dtype=np.float32)
	orig_hi_ycbcr[:,:] = sc.imresize(orig_hi_ycbcr_1[:,:], (orig_gt.shape[0] * scale, orig_gt.shape[1] * scale),
                                       interp='bicubic', mode='F')
        orig_hi_y = orig_hi_ycbcr    
    
    net_out = patch_VDSR(net,orig_hi_y,batch)



    if mode=='RGB':
        vdsr_out = orig_hi_ycbcr
	vdsr_out[:,:,0] = net_out
        vdsr_out = ycbcr2rgb(vdsr_out)*1.0
    else:
        vdsr_out = net_out * 1.0
	
    #fix for discoloration/washing out
    vdsr_out[vdsr_out < -1] = -1;
    vdsr_out[vdsr_out > 1.0] = 1.0
    #vdsr_out = vdsr_out.astype(np.uint8)
    filename_out = os.path.join(out_path, os.path.splitext(os.path.basename(file))[0] + '__VD.png')
    print(filename_out)
    vdsr_out = img_as_uint(vdsr_out)
    io.imsave(filename_out,vdsr_out)


if __name__ == '__main__':

    if(gpu_mode==1):
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    net = caffe.Net(model, weights, caffe.TEST)
    batch = batch_size
    #Read in Image
    with open(im_path_file) as f:
    	im_paths = f.readlines()
	for im_path in im_paths:
	 	process(im_path.rstrip('\n'), net, scale,batch)
    


    
