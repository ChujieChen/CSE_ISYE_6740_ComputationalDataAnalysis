% eigenfaces
clear 
close all
clc

% read the file
faces = dir('training/*.bmp');
Totalimage = length(faces);


data_raw=[];
irow=65;
icol=65;

%Chosen std and mean. 
%It can be any number that it is close to the std and mean of most of the images.
um=100;
ustd=80;

for i = 1 : Totalimage
   image = imread(strcat('training/', faces(i).name));
   %  image processoing
   I=rgb2gray(image);  % chage color to gray
   I=imresize(I,[irow,icol]);
   temp=double(reshape(I',irow*icol,1));  
   data_raw=[data_raw temp]; % "data_raw" is our orignial dataset size: 4875*172 
end


%Here we change the mean and std of all images. We normalize all images.
%This is done to reduce the error due to lighting conditions.
for i=1:size(data_raw,2)
    temp=double(data_raw(:,i));
    m=mean(temp);
    st=std(temp);
    data(:,i)=(temp-m)*ustd/st+um;
end


% obtain the mean image
mean_face=mean(data,2);
temp_mean=reshape(mean_face,icol, irow);
temp_mean=temp_mean';
figure
imshow(temp_mean/255);
title('Mean Face','fontsize',14)


%reduce the mean
data_C = bsxfun(@minus, data, mean_face); 

%here, instead of calculating eigenvectors of data_C*data_C', 
% we calculate the eigenvectors of data_C'*data_C

L=data_C'*data_C;
[vv dd]=eig(L);
% Sort and eliminate those whose eigenvalue is zero
v=[];
d=[];
for i=1:size(vv,2) 
        v=[v vv(:,i)];
        d=[d dd(i,i)]; 
 end
 
 %sort,  will return an ascending sequence
 [B index]=sort(d);
 ind=zeros(size(index));
 dtemp=zeros(size(index));
 vtemp=zeros(size(v));
 len=length(index);
 for i=1:len
    dtemp(i)=B(len+1-i);
    ind(i)=len+1-index(i);
    vtemp(:,ind(i))=v(:,i);
 end
 d=dtemp;
 v=vtemp;

 
%Normalization of eigenvectors
 for i=1:size(v,2)       %access each column
   kk=v(:,i);
   temp=sqrt(sum(kk.^2));
   v(:,i)=v(:,i)./temp;
end

%Eigenvectors of C matrix
u=[];
for i=1:size(v,2)
    temp=sqrt(d(i));
    u=[u (data_C*v(:,i))./temp];
end

%Normalization of eigenvectors
for i=1:size(u,2)
   kk=u(:,i);
   temp=sqrt(sum(kk.^2));
   u(:,i)=u(:,i)./temp;
end
 

% show the first K=6 eigenfaces
figure
K=6;
for i=1:K
  subplot(2,3,i)% eigenface
  temp=reshape(u(:,i),icol, irow);
  temp=temp';
  temp=histeq(temp,255);
  imshow(temp);
  if i==2
        title('Eigenfaces','fontsize',18)
    end
end





% Find the weight of each face in the training set.
omega = [];
for h=1:size(data_C,2)
    WW=[];    
    for i=1:size(u,2)
        t = u(:,i)';    
        WeightOfImage = dot(t,data_C(:,h)');
        WW = [WW; WeightOfImage];
    end
    omega = [omega WW];
end



% Acquire new image

% Professor and TAs image names
% The testing file includes the following images
%Le.bmp
%Amir.bmp
%Bo.bmp
%Kaushik.bmp
%Shuang.bmp




Testing= dir('testing/*.bmp');
InputImage = input('Please enter the name of the image and its extension \n','s');
InputImage = imread(strcat('testing/',InputImage));

InputImage=rgb2gray(InputImage);  % chage color to gray
InputImage=imresize(InputImage,[irow,icol]);

figure
subplot(1,2,1)
imshow(InputImage)
title('Input image','fontsize',18)
InImage=reshape(double(InputImage)',irow*icol,1);  
temp=InImage;
me=mean(temp);
st=std(temp);
temp=(temp-me)*ustd/st+um;
NormImage = temp;
Difference = temp-mean_face;


p = [];
aa=size(u,2);
for i = 1:aa
    pare = dot(NormImage,u(:,i));
    p = [p; pare];
end
ReshapedImage = mean_face+ u(:,1:aa)*p;    %m is the mean image, u is the eigenvector
ReshapedImage = reshape(ReshapedImage,icol,irow);
ReshapedImage = ReshapedImage';
%show the reconstructed image.
subplot(1,2,2)
imshow(ReshapedImage); 
title('Reconstructed image','fontsize',18)

InImWeight = [];
for i=1:size(u,2)
    t = u(:,i)';
    WeightOfInputImage = dot(t,Difference');
    InImWeight = [InImWeight; WeightOfInputImage];
end

ll = 1:Totalimage;
figure
subplot(1,2,1)
stem(ll,InImWeight)
title('Weight of Input Face','fontsize',14)

% Find Euclidean distance
e=[];
for i=1:size(omega,2)
    q = omega(:,i);
    DiffWeight = InImWeight-q;
    mag = norm(DiffWeight);
    e = [e mag];
end

kk = 1:size(e,2);
subplot(1,2,2)
stem(kk,e)
title('Eucledian distance of input image','fontsize',14)

MaximumValue=max(e)
MinimumValue=min(e)







