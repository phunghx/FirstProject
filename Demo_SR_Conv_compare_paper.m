close all;
clear all;

run matconvnet/matlab/vl_setupnn;

addpath('utils')

load('VDSR_Official.mat');
%load('VDSR_Adam.mat');
use_cascade = 0;

use_gpu = 1;
up_scale = 2;
shave = 0;
%im_gt = imread('./Data/Set5/butterfly_GT.bmp');
%im_gt = imread('/data3/FreeWork/caffeVDSR/Test/Set14/flowers.bmp');
INPUT = '/data3/FreeWork/caffeVDSR/Test/';

%INPUT = '/data3/FreeWork/caffe-vdsr/Test/Data/';
%PYTHON = '/data3/FreeWork/caffeVDSR/VDSR/Results_4x/';
PYTHON = '/data3/FreeWork/result_paper/VDSR/';
%folders = {'Set5','Set14','Urban100','B100'};
folders = {'Set5'};
%OUTPUT = '/data3/FreeWork/caffeVDSR/VDSR/Result_Mat_4x/';
%mkdir(OUTPUT);
%cd(OUTPUT);
%for i=1:size(folders,2)
%    mkdir(folders{i});
%end
if use_gpu
    for i = 1:20
        model.weight{i} = gpuArray(model.weight{i});
        model.bias{i} = gpuArray(model.bias{i});
    end
end

for i_folder=1:size(folders,2)
    psnr_py = 0;
    psnr_sr = 0;
    ssim_py = 0;
    ssim_sr = 0;
    total = 0;
    disp(folders{i_folder});
    cd([INPUT folders{i_folder}]);
    filenames = dir('*.*');
    for i_file=3:size(filenames,1),
        %disp(filenames(i_file).name);
        im_gt = imread([filenames(i_file).folder '/' filenames(i_file).name]);
    

        im_gt = modcrop(im_gt,up_scale);
        im_l = imresize(im_gt,1/up_scale,'bicubic');
        im_gt = double(im_gt);
        im_l  = double(im_l) / 255.0;

        [H,W,C] = size(im_l);
        if C == 3
            im_l_ycbcr = rgb2ycbcr(im_l);
        else
            im_l_ycbcr = im_l;
        end
        im_l_y = im_l_ycbcr(:,:,1);
        if use_gpu
            im_l_y = gpuArray(im_l_y);
        end
        %tic;
        im_h_y = VDSR_Matconvnet(im_l_y, model,up_scale,use_cascade);
        %toc;
        if use_gpu
            im_h_y = gather(im_h_y);
        end
        im_h_y = im_h_y * 255;
        im_h_ycbcr = imresize(im_l_ycbcr,up_scale,'bicubic');
        if C == 3
            im_b = ycbcr2rgb(im_h_ycbcr) * 255.0;
            im_h_ycbcr(:,:,1) = im_h_y / 255.0;
            im_h  = ycbcr2rgb(im_h_ycbcr) * 255.0;
        else
            im_h = im_h_y;
            im_b = im_h_ycbcr * 255.0;
        end

        %figure;imshow(uint8(im_b));title('Bicubic Interpolation');
        %figure;imshow(uint8(im_h));title('SR Reconstruction');
        %figure;imshow(uint8(im_gt));title('Origin');

        if shave == 1;
            shave_border = round(up_scale);
        else
            shave_border = 0;
        end

        [sr_psnr, sr_ssim] = compute_psnr(double(uint8(im_h)),im_gt,shave_border);
        
        [bi_psnr, bi_ssim] = compute_psnr(im_b,im_gt,shave_border);
        pythonFile = strsplit(filenames(i_file).name,'.');
        %imwrite(uint8(im_h), [OUTPUT folders{i_folder} '/' pythonFile{1} '__VD.png']);
        pythonFile = [pythonFile{1} '.png'];
        
        
        %img3 = double(imread([PYTHON folders{i_folder} '/' pythonFile]));            
        img3 = double(imread([PYTHON folders{i_folder} '/' pythonFile]));
        [py_psnr, py_ssim] = compute_psnr(img3,im_gt,shave_border);
        %fprintf('sr_psnr: %f dB %f\n',sr_psnr,sr_ssim);
        %fprintf('py_psnr: %f dB %f\n',py_psnr,py_ssim);
        psnr_py = psnr_py + py_psnr;
        psnr_sr = psnr_sr + sr_psnr;
        
        ssim_py = ssim_py + py_ssim;
        ssim_sr = ssim_sr + sr_ssim;
        total = total + 1;
    end
    fprintf('===========Final Report============\n');
    fprintf('MatLab: psnr: %f dB, ssim: %f\n',psnr_sr/total,ssim_sr/total);
    fprintf('Python: psnr: %f dB, ssim: %f\n',psnr_py/total,ssim_py/total);
end


