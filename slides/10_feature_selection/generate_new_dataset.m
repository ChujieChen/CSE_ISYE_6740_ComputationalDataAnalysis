% Build new boys and girls data set

clear;
close all;
clc;

% read the file
boyFaces = dir('boys/*.bmp');
totalBoyImage = length(boyFaces);

irow=65;
icol=65;

boyRaw = zeros([irow*icol, totalBoyImage]);

for i = 1 : totalBoyImage
   image = imread(strcat('boys/', boyFaces(i).name));
   %  image processoing
   I=rgb2gray(image);  % chage color to gray
   I=imresize(I,[irow,icol]);
   temp=double(reshape(I,irow*icol,1));   
   boyRaw(:,i)=temp; 
end

girlFaces = dir('girls/*.bmp');
totalGirlImage = length(girlFaces);

girlRaw = zeros([irow*icol, totalGirlImage]);

for i = 1 : totalGirlImage
   image = imread(strcat('girls/', girlFaces(i).name));
   %image processoing
   I=rgb2gray(image);  % chage color to gray
   I=imresize(I,[irow,icol]);
   temp=double(reshape(I,irow*icol,1));   
   girlRaw(:,i)=temp; 
end
    
