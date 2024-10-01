close all;
clear all;
clc;

function out=MaxRGB(im)
% Image should be normalized to 0-1 归一化

R_max = max(max(im(:,:,1)));
G_max = max(max(im(:,:,2)));
B_max = max(max(im(:,:,3)));
Max = max(im(:));

k = [R_max G_max B_max]./Max;

for i=1:3
    out(:,:,i) = im(:,:,i)/k(i);
end

end

function out=GrayWorld(im, flag)
% Image should be normalized to 0-1 归一化

R_avg = mean2(im(:,:,1));
G_avg = mean2(im(:,:,2));
B_avg = mean2(im(:,:,3));

if ~exist('flag','var')||flag==0
    Avg = 0.5;
elseif flag==1
    Avg = mean2(im);
else
    Avg = G_avg;
end
    
k = [R_avg G_avg B_avg]./Avg;

for i=1:3
out(:,:,i) = im(:,:,i)/k(i);
out(:,:,i) = min(out(:,:,i),1);%处理一下有可能超出1的值，直接设1
end

end

function out=sharping(img)
    sigma = 20;
    Igauss = img;
    N = 30;
    for iter=1: N
       Igauss =  imgaussfilt(Igauss,sigma);
       Igauss = min(img, Igauss);
    end
    
    gain = 1; %in the paper is not mentioned, but sometimes gain <1 is better. 
    Norm = (img-gain*Igauss);
    %Norm
    for n = 1:3
       Norm(:,:,n) = histeq(Norm(:,:,n)); 
    end
    Isharp = (img + Norm)/2;
    out = Isharp;
end


function [out1, out2]=comb(IS, IG)
    %% weights calculation
    
    % Lapacian contrast weight 
    Isharp_lab = rgb2lab(IS);
    Igamma_lab = rgb2lab(IG);
    
    % input1
    R1 = double(Isharp_lab(:, :, 1)) / 255;
    % calculate laplacian contrast weight
    WC1 = sqrt((((IS(:,:,1)) - (R1)).^2 + ...
                ((IS(:,:,2)) - (R1)).^2 + ...
                ((IS(:,:,3)) - (R1)).^2) / 3);
    % calculate the saliency weight
    WS1 = saliency_detection(IS);
    WS1 = WS1/max(WS1,[],'all');
    % calculate the saturation weight
    
    WSAT1 = sqrt(1/3*((IS(:,:,1)-R1).^2+(IS(:,:,2)-R1).^2+(IS(:,:,3)-R1).^2));
    
    
    %figure('name', 'Image 1 weights');
    %imshow([WC1 , WS1, WSAT1]);
    
    
    % input2
    R2 = double(Igamma_lab(:, :, 1)) / 255;
    % calculate laplacian contrast weight
    WC2 = sqrt((((IG(:,:,1)) - (R2)).^2 + ...
                ((IG(:,:,2)) - (R2)).^2 + ...
                ((IG(:,:,3)) - (R2)).^2) / 3);
    % calculate the saliency weight
    WS2 = saliency_detection(IG);
    WS2 = WS2/max(WS2,[],'all');
    
    % calculate the saturation weight
    WSAT2 = sqrt(1/3*((IG(:,:,1)-R1).^2+(IG(:,:,2)-R1).^2+(IG(:,:,3)-R1).^2));
    
    %figure('name', 'Image 2 weights');
    %imshow([WC2 , WS2, WSAT2]);
    
    % calculate the normalized weight
    W1 = (WC1 + WS1 + WSAT1+0.1) ./ ...
         (WC1 + WS1 + WSAT1 + WC2 + WS2 + WSAT2+0.2);
    W2 = (WC2 + WS2 + WSAT2+0.1) ./ ...
         (WC1 + WS1 + WSAT1 + WC2 + WS2 + WSAT2+0.2);
     
     
    %% Naive fusion
    R = W1.*IS+W2.*IG;
    %figure('name', 'Naive Fusion');
    %imshow([IS, IG, R]);
    out1 = R;
    
    %% Multi scale fusion.
    img1 = IS;
    img2 = IG;
    
    % calculate the gaussian pyramid
    level = 5;
    Weight1 = gaussian_pyramid(W1, level);
    Weight2 = gaussian_pyramid(W2, level);
    
    % calculate the laplacian pyramid
    % input1
    R1 = laplacian_pyramid(IS(:, :, 1), level);
    G1 = laplacian_pyramid(IS(:, :, 2), level);
    B1 = laplacian_pyramid(IS(:, :, 3), level);
    % input2
    R2 = laplacian_pyramid(IG(:, :, 1), level);
    G2 = laplacian_pyramid(IG(:, :, 2), level);
    B2 = laplacian_pyramid(IG(:, :, 3), level);
    
    % fusion
    for k = 1 : level
       Rr{k} = Weight1{k} .* R1{k} + Weight2{k} .* R2{k};
       Rg{k} = Weight1{k} .* G1{k} + Weight2{k} .* G2{k};
       Rb{k} = Weight1{k} .* B1{k} + Weight2{k} .* B2{k};
    end
    
    % reconstruct & output
    R = pyramid_reconstruct(Rr);
    G = pyramid_reconstruct(Rg);
    B = pyramid_reconstruct(Rb);
    fusion = cat(3, R, G, B);
    out2 = fusion;
end


rgbImage = double(imread('TestPhoto/photo6jpg.jpg'))/255;
grayImage = rgb2gray(rgbImage);

chanelR = rgbImage(:, :, 1);
chanelG = rgbImage(:, :, 2);
chanelB = rgbImage(:, :, 3);
%imwrite([chanelR, chanelG, chanelB], "./photo/chanel3.jpg");
meanR = mean(chanelR, "all");
meanG = mean(chanelG, "all");
meanB = mean(chanelB, "all");

%% 直接白平衡未做補償的結果(灰度世界)
IW = cat(3, chanelR, chanelG, chanelB);
IW_lin = rgb2lin(IW);
IW_lin = GrayWorld(IW_lin);
IWb = lin2rgb(IW_lin);

%% MaxRGB
IM = cat(3, chanelR, chanelG, chanelB);
IM_lin = rgb2lin(IM);

IM_lin = MaxRGB(IM_lin);

IMb = lin2rgb(IM_lin);

I_lin = rgb2lin(IM);
percentiles = 5;
illuminant = illumgray(I_lin,percentiles);
I_lin = chromadapt(I_lin,illuminant,'ColorSpace','linear-rgb');
Iwb = lin2rgb(I_lin);
%figure("Name", "Only White Balance");
%imshow([rgbImage, IMb, IWb]);
%imwrite([rgbImage, IMb, IWb], "./photo/OnlyWhite2.jpg");

%% 顏色補償
alpha = 0.1;
chanelRc = chanelR + alpha*(meanG - meanR);
alpha = 0; % 0 does not compensates blue channel. 
chanelBc = chanelB + alpha*(meanG - meanB);

%% 有補償的結果(灰度世界)
IWc = cat(3, chanelRc, chanelG, chanelBc);
IWc_lin = rgb2lin(IWc);
IWc_lin = GrayWorld(IWc_lin);
IWbc = lin2rgb(IWc_lin);

%% MaxRGB
IMc = cat(3, chanelRc, chanelG, chanelBc);
IMc_lin = rgb2lin(IMc);

IMc_lin = MaxRGB(IMc_lin);

IMbc = lin2rgb(IMc_lin);

%% author
I_linc = rgb2lin(IMc);
percentiles = 10;
illuminant = illumgray(I_linc,percentiles);
I_linc = chromadapt(I_linc,illuminant,'ColorSpace','linear-rgb');
Iwbc = lin2rgb(I_linc);


%figure("Name", "Color compensation + White Balance");
%imshow([rgbImage,IWb, histeq(IWbc)]);
%imwrite([rgbImage,IWb, IWbc], "./photo/Color&White1.jpg");

%figure("Name", "Color compensation + Chromadapt");
%imshow([rgbImage, histeq(IWbc), Iwbc]);
%imwrite([rgbImage,histeq(IWbc), Iwbc], "./photo/Color&Chromadapt1.jpg");


Igamma = imadjust(Iwbc,[],[],2);
Isharp = sharping(Iwbc);
%imshow([Iwbc, Igamma, Isharp]);
%imwrite([Iwbc, Igamma, Isharp], "./photo/Gamma&Sharp3.jpg");


%IgammaW = imadjust(IWbc, [], [], 2);
%imshow(IgammaW);
%IsharpW = sharping(IWbc); 

%IgammaM = imadjust(IMbc, [], [], 2);
%IsharpM = sharping(IMbc); 

[out1, out2]= comb(Isharp, Igamma);
%figure('Name','out1');
%imshow(out1);
figure('Name','result');
imshow([rgbImage,out2]);

%[out1, out2]= comb(IsharpW, IgammaW);
%figure('Name','out1W');
%imshow(out1);
%figure('Name','out2W');
%imshow(out2);

%[out1, out2]= comb(IsharpM, IgammaM);
%figure('Name','out1M');
%imshow(out1);
%figure('Name','out2M');
%imshow(out2);
