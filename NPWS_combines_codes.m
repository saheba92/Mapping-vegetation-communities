%% FOR SHARING WITH NPWS %%
clear all
clc
close all

%% Please cite as following if using:
%APA style: 
%Bhatnagar, S., Gill, L., Regan, S., Naughton, O., Johnston, P., Waldren, S., & Ghosh, B. (2020). Mapping vegetation communities inside wetlands using Sentinel-2 imagery in Ireland. International Journal of Applied Earth Observation and Geoinformation, 88, 102083.
%bibtex
% @article{bhatnagar2020mapping,
%   title={Mapping vegetation communities inside wetlands using Sentinel-2 imagery in Ireland},
%   author={Bhatnagar, Saheba and Gill, Laurence and Regan, Shane and Naughton, Owen and Johnston, Paul and Waldren, Steve and Ghosh, Bidisha},
%   journal={International Journal of Applied Earth Observation and Geoinformation},
%   volume={88},
%   pages={102083},
%   year={2020},
%   publisher={Elsevier}
% }

%% LOAD THE DATA %%

[path,user_cance]=imgetfile();
if user_cance
    msgbox(sprintf('Error'),'Error','Error');
    return
end

%% READ THE IMAGE %%

[img R] = geotiffread(path); %imread('name.jpg');
info = geotiffinfo(path); % comment if not spatial image

[x y z] = size(img);
Ib = double(reshape(img,x*y,z));

% to view the image
figure; imagesc(img(:,:,1)); axis off; colormap jet

%% Masking image - comment for not masking
h = imfreehand; %draw something
M = ~h.createMask();
I(M) = 0;
imshow(I,[]);
BW = 1-M;
figure; imshow(BW,[]);
maskedImage1 = img ;maskedImage1(repmat(~BW,[1 1 1])) = 0; 

%% choose a rectangle %%

%%
for i = 1:z
    maskedImage1(repmat(~BW,[1 1 i])) = 0;
    m2(:,:,i) = maskedImage1(:,:,i);
end

Ib = double(reshape(m2,x*y,z));
%% 

prompt = 'Supervised (1) or Unsupervised (2) ';
k = input(prompt)

close all

%%
if k ==1
%% NORMALISE THE IMAGE - OPTIONAL
I=[]; X=[];

for i =1:z
    mi = min(Ib(:,i));
    ma = max(Ib(:,i));
    %     mi1= min(img(:,i));
    %     ma1= max(img(:,i));
    diff = ma-mi;
    I(:,i) = (Ib(:,i)-mi)/(diff);
end
figure; imagesc(m2(:,:,1)); axis off; colormap gray
A_res = I;

prompt = 'Select ROI for class 1; double click to stop selecting points'
[xi,yi] = getpts;
r = ceil(xi); c = ceil(yi);
%% finding 16 neighbours of all the pixels(mask) 
nr = 2; % it is a 4x4 neighbour
% ne = 16; %the height and width of the neighbourhood rectangle/sqaure
[ul] = [r c] ;%upper left neighbour
xx = ul(:,2);
yy = ul(:,1);

temp=1:x*y;  %(multiplied dimensions of image)
temp=temp';
temp=reshape(temp,x,y);
q=0;
for i=1:size(ul,1)
	for j=xx(i)-2:xx(i)+2   %for 16 neighbours --> xx(i)-1:xx(i)+2 (same for y)
	for l=yy(i)-2:yy(i)+2
q=q+1;
mask(q)=temp(j,l);
	end
	end
end
A1 = A_res(mask,:);

TR1 = zeros(size(A1,1),z+1);
TR1(:,1:end-1) =  A1;
TR1 (:,end) = 1;

hold on;

prompt = 'Select ROI for class 2'
[xi,yi] = getpts;
r = ceil(xi); c = ceil(yi);
[ul] = [r c] ;%upper left neighbour
xx = ul(:,2);
yy = ul(:,1);

temp=1:x*y;  %(multiplied dimensions of image)
temp=temp';
temp=reshape(temp,x,y);
q=0;
for i=1:size(ul,1)
	for j=xx(i)-2:xx(i)+2   
	for l=yy(i)-2:yy(i)+2
q=q+1;
mask(q)=temp(j,l);
	end
	end
end
A2 = A_res(mask,:);

TR2 = zeros(size(A2,1),z+1);
TR2(:,1:end-1) =  A2;
TR2 (:,end) = 2;

hold on;

prompt = 'Select ROI for class 3'
[xi,yi] = getpts;
r = ceil(xi); c = ceil(yi);
[ul] = [r c] ;%upper left neighbour
xx = ul(:,2);
yy = ul(:,1);

temp=1:x*y;  %(multiplied dimensions of image)
temp=temp';
temp=reshape(temp,x,y);
q=0;
for i=1:size(ul,1)
	for j=xx(i)-2:xx(i)+2   
	for l=yy(i)-2:yy(i)+2
q=q+1;
mask(q)=temp(j,l);
	end
	end
end
A3 = A_res(mask,:);

TR3 = zeros(size(A3,1),z+1);
TR3(:,1:end-1) =  A3;
TR3 (:,end) = 3;

hold on;

prompt = 'Select ROI for class 4'
[xi,yi] = getpts;
r = ceil(xi); c = ceil(yi);
[ul] = [r c] ;%upper left neighbour
xx = ul(:,2);
yy = ul(:,1);

temp=1:x*y;  %(multiplied dimensions of image)
temp=temp';
temp=reshape(temp,x,y);
q=0;
for i=1:size(ul,1)
	for j=xx(i)-2:xx(i)+2   
	for l=yy(i)-2:yy(i)+2
q=q+1;
mask(q)=temp(j,l);
	end
	end
end
A4 = A_res(mask,:);

TR4 = zeros(size(A4,1),z+1);
TR4(:,1:end-1) =  A4;
TR4 (:,end) = 4;

prompt = 'Select ROI for class 5'
[xi,yi] = getpts;
r = ceil(xi); c = ceil(yi);
[ul] = [r c] ;%upper left neighbour
xx = ul(:,2);
yy = ul(:,1);

temp=1:x*y;  %(multiplied dimensions of image)
temp=temp';
temp=reshape(temp,x,y);
q=0;
for i=1:size(ul,1)
	for j=xx(i)-2:xx(i)+2   
	for l=yy(i)-2:yy(i)+2
q=q+1;
mask(q)=temp(j,l);
	end
	end
end
A5 = A_res(mask,:);

TR5 = zeros(size(A5,1),z+1);
TR5(:,1:end-1) =  A5;
TR5 (:,end) = 5;

%% copy paste the same code for adding more classes 
prompt = 'Select ROI for class 6'
[xi,yi] = getpts;
r = ceil(xi); c = ceil(yi);
[ul] = [r c] ;%upper left neighbour
xx = ul(:,2);
yy = ul(:,1);

temp=1:x*y;  %(multiplied dimensions of image)
temp=temp';
temp=reshape(temp,x,y);
q=0;
for i=1:size(ul,1)
	for j=xx(i)-2:xx(i)+2   
	for l=yy(i)-2:yy(i)+2
q=q+1;
mask(q)=temp(j,l);
	end
	end
end
A6 = A_res(mask,:);

TR6 = zeros(size(A6,1),z+1);
TR6(:,1:end-1) =  A6;
TR6 (:,end) = 6;
%%
% prompt = 'Select ROI for class 7'
% [xi,yi] = getpts;
% r = ceil(xi); c = ceil(yi);
% [ul] = [r c] ;%upper left neighbour
% xx = ul(:,2);
% yy = ul(:,1);
% 
% temp=1:x*y;  %(multiplied dimensions of image)
% temp=temp';
% temp=reshape(temp,x,y);
% q=0;
% for i=1:size(ul,1)
% 	for j=xx(i)-2:xx(i)+2   
% 	for l=yy(i)-2:yy(i)+2
% q=q+1;
% mask(q)=temp(j,l);
% 	end
% 	end
% end
% A4 = A_res(mask,:);
% 
% TR4 = zeros(size(A4,1),z+1);
% TR4(:,1:end-1) =  A4;
% TR4 (:,end) = 4;
%%
TRAIN = [TR1;TR2;TR3;TR4;TR5;TR6]; %; BTR1; BTR2; BTR3; BTR4; BTR5; BTR6];%;TR7;TR8;TR9];
assignin('base','TRAIN',TRAIN) %for saving TRAIN variable in workspace
%% TRAINING THE DATA USING ENSEMBLE LEARNER %%

rng('default')
t = templateTree('Reproducible',true);
Mdl = fitcensemble(TRAIN(:,1:end-1),TRAIN(:,end),'Method','Bag');

view(Mdl.Trained{1},'Mode','graph')

partitionedModel = crossval(Mdl, 'KFold', 5);
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError')

[yfit p] = predict(Mdl,A_res);
% [yfit p] = trainedClassifier.predictFcn(A_res);
%[x y z] = size(A);
yfit_res = reshape(yfit,x,y,1);
figure;imagesc(yfit_res); colormap(jet); axis off; title('Pixel-based');
filename='supervised_file1';geotiffwrite(filename,yfit_res,R, 'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);

C = imfuse(yfit_res,img(:,:,1));
figure; imagesc(C);  axis off; colormap(jet); title('Overlayed Image');

%% YOU WOULD NEED VISUAL STUDIO %%
%% PLEASE COMMENT IF VISUAL STUDIO IS NOT INSTALLED %%
%% GC MEX INSTALLER TO BE INSTALLED AND ACTIVATED %% 
%addpath H:/Clara
no_classes = max(yfit);
p=p';
Dc = reshape(log(p+eps)',x, y ,(no_classes));
Sc = ones((no_classes)) - eye((no_classes));
[gch] = GraphCut('open', -Dc,0.7*Sc);
[gch labels] = GraphCut('expand', gch);
[gch] = GraphCut('close', gch);
figure;
subplot(1,2,1); imagesc(yfit_res); colormap(jet); axis off; title('Pixel-based');
subplot(1,2,2); imagesc(labels); colormap(jet); axis off; title('Segment-based');
filename='supervised_file2';geotiffwrite(filename,labels,R, 'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);

C = imfuse(labels,img(:,:,1));
figure; imagesc(C) ; axis off; colormap(jet); title('Overlayed Image');

else
   prompt = 'How many clusters to be formed including background'
n = input(prompt)

idx = kmeans(Ib,n);
kidx = reshape(idx,x,y,1);

figure; imagesc(kidx); axis off; colormap jet; colorbar; title('K means');

filename='unsupervised_file1';geotiffwrite(filename,kidx,R, 'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);

C = imfuse(kidx,img(:,:,1));
figure; imagesc(C); axis off; colormap(jet); title('Overlayed Image');
end

%%
