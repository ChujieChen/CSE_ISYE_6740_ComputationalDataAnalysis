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

for i = 1 : Totalimage
   image = imread(strcat('training/', faces(i).name));
   %  image processoing
   I=rgb2gray(image);  % chage color to gray
   I=imresize(I,[irow,icol]);
   temp=reshape(I,irow*icol,1);  
   data_raw=[data_raw temp]; % "data_raw" is our orignial dataset size: 4875*172 
end

figure; 
show_image(data_raw', irow, icol); 
drawnow; 
saveas(gcf, 'allfaces', 'png'); 

% return; 

data = double(data_raw); 

% obtain the mean image
mean_face=mean(data,2);
temp_mean=reshape(mean_face,icol, irow);
figure
imshow(uint8(temp_mean));
title('Mean Face','fontsize',14)
drawnow; 
saveas(gcf, 'meanfaces', 'png'); 

% return; 

%reduce the mean
data_C = bsxfun(@minus, data, mean_face); 

% % note that left singular vector of data_C is equal to eigenvector of
% data_C * data_C'; In this problem, svd is much faster. 
% L=data_C*data_C';
% [vv dd]=eig(L);
[u, s, v]=svd(data_C./sqrt(Totalimage)); 
ss = diag(s); 

figure; 
subplot(1,2,1); 
plot(ss.^2./sum(ss.^2));
title('Eigenspectrum'); 
subplot(1,2,2); 
plot(cumsum(ss.^2./sum(ss.^2))); 
title('Cumulative Sum of Eigenspectrum'); 
drawnow; 
 
% show the first K=6 eigenfaces
figure
K=20;
for i=1:K
  subplot(4,5,i)% eigenface
  temp=reshape(u(:,i),icol, irow);
  temp = 255 * (temp - min(temp(:))) ./ (max(temp(:)) - min(temp(:))); 
  imshow(uint8(temp));
  if (i==1)
    title('Eigenfaces'); 
  end
end

saveas(gcf, 'eigenfaces', 'png'); 

% pick 60 components to preserve about 95% variance of the data; 
component_no = 100; 
omega = u(:,1:component_no)' * data_C;

selected_i = 95; 
reconstructed_I = mean_face; 
temp_image = []; 

input('press any key to continue\n'); 

figure; 
for i = 1:component_no
  reconstructed_I = reconstructed_I + omega(i,selected_i) * u(:,i); 
  temp_image = [temp_image, 255 * (reconstructed_I - min(reconstructed_I)) ...
    ./ (max(reconstructed_I) - min(reconstructed_I))]; 
  
  imshow(uint8(reshape(temp_image(:,i), irow, icol))); 
  title([int2str(i), ' components']); 
  drawnow; 
  
  if (mod(i, 20) == 0)
    input('press any key to continue\n'); 
  end  
end

% input('press any key to continue\n');

figure; 
show_image(temp_image', irow, icol);
drawnow; 
input('press any key to continue\n');

%% Acquire new image

% Professor and TAs image names
% The testing file includes the following images
%Le.bmp
%Amir.bmp
%Bo.bmp
%Kaushik.bmp

return

Testing= dir('testing/*.bmp');
% InputImage = input('Please enter the name of the image and its extension \n','s');
InputImage = 'bo.bmp'; 
InputImage = imread(strcat('testing/',InputImage));

InputImage=rgb2gray(InputImage);  % chage color to gray
InputImage=imresize(InputImage, [irow,icol]);
figure; 
imshow(InputImage); 
drawnow; 
input('press any key to continue\n');
newdata = double(InputImage(:)); 

component_no = Totalimage; 
omega = u' * (newdata - mean_face);  

reconstructed_I = mean_face; 
temp_image = []; 

figure; 
for i = 1:component_no
  reconstructed_I = reconstructed_I + omega(i) * u(:,i); 
  temp_image = [temp_image, 255 * (reconstructed_I - min(reconstructed_I)) ...
    ./ (max(reconstructed_I) - min(reconstructed_I))]; 
  
  imshow(uint8(reshape(temp_image(:,i), irow, icol))); 
  title([int2str(i), ' components']); 
  drawnow; 
  
end
