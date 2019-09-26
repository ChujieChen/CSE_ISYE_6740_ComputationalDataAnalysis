close all

 load 'faces'
%We plot samples of the face data set.

faceW = 64; 
faceH = 64; 
numPerLine = 16; 
ShowLine = 8; 

Y = zeros(faceH*ShowLine,faceW*numPerLine); 
f=double(images);
for i=0:ShowLine-1 
  	for j=0:numPerLine-1 
    	 Y(i*faceH+1:(i+1)*faceH,j*faceW+1:(j+1)*faceW) = reshape(images(:,i*numPerLine+j+1),[faceH,faceW]); 
  	end 
end 

figure()
imagesc(Y);
colormap(gray);
axis off;
axis equal;

%% Isomap

% This is how we do the Isomap.We have commented the code and load the
% previously run version.The reason is that each time we run the code, the
% result may be different. As we are assigning images to the points in 2D
% (See the result) manually, we can't run from the begining.

% x = f;
% 
% m = size(x, 2); 
% 
% G = sqrt(sum(x.^2,1)'*ones(1,m) + ones(m,1)*sum(x.^2,1) - 2*(x'*x));
%  e = 0.6*median(G(:));  
%  G(G>e) = 0; 
% sG = sum(G, 1); 
% 
% % get rid of Inf distance for simplicity; 
% i = find(sG == 0); 
% idx = setdiff((1:m), i); 
% G = G(idx,idx); 
% m = size(G, 1); 
% 
% spy(sparse(G)); 
% drawnow; 
% 
% D = graphallshortestpaths(sparse(G), 'directed', 'false');
% % D(D>10^20)=0;
% D2 = D.^2; 
% 
% H = eye(m) - ones(m,1)*ones(m,1)'./m; 
% 
% Dt = -0.5 * H * D2 * H; 
% 
% k = 10; 
% [V, S, U] = svds(Dt, k);
% 
% dim1 = V(:,1) * sqrt(S(1,1)); 
% dim2 = V(:,2) * sqrt(S(2,2)); 

%%
load 'dim1'
load 'dim2'
%%

% In this block we fit images of the digits to the points in 2D.

a=find(dim1>40 & dim1<40.2);
b=find(dim1>-46 & dim1<-45.92);
c=find(dim1>-7.2 & dim1<-7.1);
d=find(dim1>-12.1 & dim1<-11.9);
e=find(dim1>28.3 & dim1<28.34);
f=find(dim1>3.36 & dim1<3.37);
g=find(dim1>18.91 & dim1<18.93);
h=find(dim1>-27.77 & dim1<-27.75);
i=find(dim1>-28.2 & dim1<-28.1);
j=find(dim1>9.1 & dim1<9.2);
k=find(dim1>-15.2 & dim1<-15);
l=find(dim1>2.49 & dim1<2.5);
m=find(dim1>20.43 & dim1<20.45);
n=find(dim1>-19.03 & dim1<-19.01);
o=find(dim1>-2.945 & dim1<-2.94);
p=find(dim1>14.5 & dim1<14.7);
q=find(dim1>39.56 & dim1<39.58);
r=find(dim1>30.1 & dim1<30.12);
s=find(dim1>-39.56 & dim1<-39.54);

th = 0:pi/50:2*pi;
xunit_b = 1 * cos(th) + -45.98;
yunit_b = 1 * sin(th) + -3.783;

xunit_a = 1 * cos(th) + 40.1;
yunit_a = 1 * sin(th) + -21.74;
xunit_c = 1 * cos(th) + -7.17;
yunit_c = 1 * sin(th) + 10.47;

xunit_d = 1 * cos(th) + -12.01;
yunit_d = 1 * sin(th) + -26.69;

xunit_e = 1 * cos(th) + 28.32;
yunit_e = 1 * sin(th) + 12.04;


xunit_f = 1 * cos(th) + 3.361;
yunit_f = 1 * sin(th) + -6.631;
xunit_g = 1 * cos(th) + 18.92;
yunit_g = 1 * sin(th) + -20.51;
xunit_h = 1 * cos(th) + -27.76;
yunit_h = 1 * sin(th) + 4.455;
xunit_i = 1 * cos(th) + -28.14;
yunit_i = 1 * sin(th) + -15.11;
xunit_j = 1 * cos(th) + 9.127;
yunit_j = 1 * sin(th) + 8.518;
xunit_k = 1 * cos(th) + -15.1;
yunit_k = 1 * sin(th) + -7.862;
xunit_l = 1 * cos(th) + 2.498;
yunit_l = 1 * sin(th) + -24.67;
xunit_m = 1 * cos(th) + 20.44;
yunit_m = 1 * sin(th) + -4.112;

xunit_n = 1 * cos(th) + -19.02;
yunit_n = 1 * sin(th) + 14.3;
xunit_o = 1 * cos(th) + -2.943;
yunit_o = 1 * sin(th) + 21.79;
xunit_p = 1 * cos(th) + 14.6;
yunit_p = 1 * sin(th) + 18.3;
xunit_q = 1 * cos(th) + 39.57;
yunit_q = 1 * sin(th) + -7.296;
xunit_r = 1 * cos(th) + 30.11;
yunit_r = 1 * sin(th) + -17.24;
xunit_s = 1 * cos(th) + -39.55;
yunit_s = 1 * sin(th) + 5.48;



a_img=reshape(images(:,614),[64 64]);
b_img=reshape(images(:,572),[64 64]);
c_img=reshape(images(:,246),[64 64]);
d_img=reshape(images(:,653),[64 64]);
e_img=reshape(images(:,244),[64 64]);
f_img=reshape(images(:,340),[64 64]);
g_img=reshape(images(:,578),[64 64]);
h_img=reshape(images(:,358),[64 64]);
i_img=reshape(images(:,25),[64 64]);
j_img=reshape(images(:,291),[64 64]);
k_img=reshape(images(:,85),[64 64]);
l_img=reshape(images(:,594),[64 64]);
m_img=reshape(images(:,525),[64 64]);
n_img=reshape(images(:,182),[64 64]);
o_img=reshape(images(:,591),[64 64]);
p_img=reshape(images(:,198),[64 64]);
q_img=reshape(images(:,500),[64 64]);
r_img=reshape(images(:,93),[64 64]);
s_img=reshape(images(:,646),[64 64]);


figure()
scatter(dim1, dim2,18*ones(698,1),'fill');
title('isomap');
hold on
imagesc([37 43],[-25 -31],a_img);
colormap(gray)
imagesc([-48 -42],[-5 -11],b_img);
colormap(gray)
imagesc([-10 -4],[8 2],c_img);
colormap(gray)
imagesc([-15 -9],[-28 -34],d_img);
colormap(gray)
imagesc([25 31],[11 5],e_img);
colormap(gray)
imagesc([0 6],[-7.7 -13.5],f_img);
colormap(gray)
imagesc([15 21],[-21.6 -27.6],g_img);
colormap(gray)
imagesc([-30.76 -24.76],[3.3 -2.7],h_img);
colormap(gray)
imagesc([-31.14 -25.14],[-16.2 -22.2],i_img);
colormap(gray)
imagesc([6.127 12.127],[7.418 1.418],j_img);
colormap(gray)
imagesc([-18.1 -12.1],[-8.96 -14.96],k_img);
colormap(gray)
imagesc([-0.51 5.49],[-25.77 -31.77],l_img);
colormap(gray)
imagesc([17.44 23.44],[-5.21 -11.21],m_img);
colormap(gray)
imagesc([-22.02 -16.02],[13.2 7.2],n_img);
colormap(gray)
imagesc([-5.94 0.06],[20.69 14.69],o_img);
colormap(gray)
imagesc([11.6 17.6],[17.2 11.2],p_img);
colormap(gray)
imagesc([36.57 42.57],[-8.39 -14.39],q_img);
colormap(gray)
imagesc([27.11 33.11],[-18.34 -24.34],r_img);
colormap(gray);
imagesc([-42.55 -37.55],[4.38 -1.62],s_img);
colormap(gray)

plot(xunit_a, yunit_a,'red');
plot(xunit_b, yunit_b,'red');
plot(xunit_c, yunit_c,'red');
plot(xunit_d, yunit_d,'red');
plot(xunit_e, yunit_e,'red');
plot(xunit_f, yunit_f,'red');
plot(xunit_g, yunit_g,'red');
plot(xunit_h, yunit_h,'red');
plot(xunit_i, yunit_i,'red');
plot(xunit_j, yunit_j,'red');
plot(xunit_k, yunit_k,'red');
plot(xunit_l, yunit_l,'red');
plot(xunit_m, yunit_m,'red');
plot(xunit_n, yunit_n,'red');
plot(xunit_o, yunit_o,'red');
plot(xunit_p, yunit_p,'red');
plot(xunit_q, yunit_q,'red');
plot(xunit_r, yunit_r,'red');
plot(xunit_s, yunit_s,'red');

hold off 

%% PCA

xx=double(images);

center=mean(xx,2);
 sz=size(xx,2);
 x = xx - center * ones(1, sz); 
 

%%
% This is how we do PCA, We have commented the code and load the
% previously run version.The reason is that each time we run the code, the
% result may be different. As we are assigning images to the points in 2D
% (See the result) manually, we can't run from the begining.



% covariance = cov(x'); 
% % [U, S] = eig(covariance); 
% [U, S] = eigs(covariance, 3); 
% 
% % proect x to the principal direction; 
% Ux = U(:,1:2)' * x; 
%%
load 'Ux_face'

%% Fitting images to plot

th = 0:pi/50:2*pi;
a=find(Ux(1,:)>-12.88 & Ux(1,:)<-12.86);
xunit_a = 0.5 * cos(th) + -12.87;
yunit_a = 0.5 * sin(th) + 7.131;

b=find(Ux(1,:)>-7.02 & Ux(1,:)<-7);
xunit_b = 0.5 * cos(th) + -7.01;
yunit_b = 0.5 * sin(th) + 7.961;

c=find(Ux(1,:)>-2.61 & Ux(1,:)<-2.60);
xunit_c = 0.5 * cos(th) + -2.608;
yunit_c = 0.5 * sin(th) + 9.03;

d=find(Ux(1,:)>2.21 & Ux(1,:)<2.22);
xunit_d = 0.5 * cos(th) + 2.216;
yunit_d = 0.5 * sin(th) + 7.823;

e=find(Ux(1,:)>6.687 & Ux(1,:)<6.689);
xunit_e = 0.5 * cos(th) + 6.688;
yunit_e = 0.5 * sin(th) + 8.025;

f=find(Ux(1,:)>-12.68 & Ux(1,:)<-12.66);
xunit_f = 0.5 * cos(th) + -12.67;
yunit_f = 0.5 * sin(th) + -0.3996;

g=find(Ux(1,:)>-7.85 & Ux(1,:)<-7.845);
xunit_g = 0.5 * cos(th) + -7.849;
yunit_g = 0.5 * sin(th) + -0.07263;


h=find(Ux(1,:)>-2.01 & Ux(1,:)<-2);
xunit_h = 0.5 * cos(th) + -2.001;
yunit_h = 0.5 * sin(th) + -0.1126;

i=find(Ux(1,:)>2.76 & Ux(1,:)<2.762);
xunit_i= 0.5 * cos(th) + 2.761;
yunit_i = 0.5 * sin(th) + 0.07668;

j=find(Ux(1,:)>7.537& Ux(1,:)<7.539);
xunit_j= 0.5 * cos(th) + 7.538;
yunit_j = 0.5 * sin(th) + 0.3146;

k=find(Ux(1,:)>-6.45 & Ux(1,:)<-6.44);
xunit_k= 0.5 * cos(th) + -6.446;
yunit_k = 0.5 * sin(th) + -8.875;

l=find(Ux(1,:)>-3.51 & Ux(1,:)<-3.49);
xunit_l= 0.5 * cos(th) + -3.5;
yunit_l = 0.5 * sin(th) + -6.264;

m=find(Ux(1,:)>1.103 & Ux(1,:)<1.105);
xunit_m= 0.5 * cos(th) + 1.104;
yunit_m = 0.5 * sin(th) + -7.177;

n=find(Ux(1,:)>4.281 & Ux(1,:)<4.283);
xunit_n= 0.5 * cos(th) + 4.282;
yunit_n = 0.5 * sin(th) + -7.097;

o=find(Ux(1,:)>7.84 & Ux(1,:)<7.842);
xunit_o= 0.5 * cos(th) + 7.841;
yunit_o = 0.5 * sin(th) + -6.649;

p=find(Ux(1,:)>-4.7 & Ux(1,:)<-4.68);
xunit_p= 0.5 * cos(th) + -4.69;
yunit_p = 0.5 * sin(th) + -15.14;

q=find(Ux(1,:)>-0.432 & Ux(1,:)<-0.431);
xunit_q= 0.5 * cos(th) + -0.4317;
yunit_q = 0.5 * sin(th) + -13.08;

r=find(Ux(1,:)>4.28 & Ux(1,:)<4.30);
xunit_r= 0.5 * cos(th) + 4.29;
yunit_r = 0.5 * sin(th) + -13.01;

s=find(Ux(1,:)>10.06 & Ux(1,:)<10.08);
xunit_s= 0.5 * cos(th) + 10.07;
yunit_s = 0.5 * sin(th) + 4.429;

t=find(Ux(1,:)>13.63 & Ux(1,:)<13.65);
xunit_t= 0.5 * cos(th) + 13.64;
yunit_t = 0.5 * sin(th) + 3.27;

z=find(Ux(1,:)>-10.56 & Ux(1,:)<-10.54);
xunit_z= 0.5 * cos(th) + -10.55;
yunit_z = 0.5 * sin(th) + 11.3;

z1=find(Ux(1,:)>-4.927 & Ux(1,:)<-4.925);
xunit_z1= 0.5 * cos(th) + -4.926;
yunit_z1 = 0.5 * sin(th) + 12.72;

a_img=reshape(images(:,224),[64 64]);
b_img=reshape(images(:,71),[64 64]);
c_img=reshape(images(:,383),[64 64]);
d_img=reshape(images(:,338),[64 64]);
e_img=reshape(images(:,400),[64 64]);
f_img=reshape(images(:,166),[64 64]);
g_img=reshape(images(:,267),[64 64]);
h_img=reshape(images(:,215),[64 64]);
i_img=reshape(images(:,140),[64 64]);
j_img=reshape(images(:,56),[64 64]);
k_img=reshape(images(:,357),[64 64]);
l_img=reshape(images(:,157),[64 64]);
m_img=reshape(images(:,4),[64 64]);
n_img=reshape(images(:,271),[64 64]);
o_img=reshape(images(:,669),[64 64]);
p_img=reshape(images(:,490),[64 64]);
q_img=reshape(images(:,432),[64 64]);
r_img=reshape(images(:,449),[64 64]);
s_img=reshape(images(:,156),[64 64]);
t_img=reshape(images(:,502),[64 64]);
z_img=reshape(images(:,627),[64 64]);
z1_img=reshape(images(:,623),[64 64]);

figure()

scatter(Ux(1,:), Ux(2,:),18*ones(1,698),'fill');
title('PCA');
hold on
imagesc([-11.87 -13.87],[6.531 3.531],a_img);
colormap(gray)
plot(xunit_a, yunit_a,'red');

imagesc([-6.01 -8.01],[7.361 4.361],b_img);
colormap(gray)
plot(xunit_b, yunit_b,'red');


imagesc([-1.608 -3.608],[8.03 5.03],c_img);
colormap(gray)
plot(xunit_c, yunit_c,'red');


imagesc([1.216 3.216],[7.223 4.223],d_img);
colormap(gray)
plot(xunit_d, yunit_d,'red');

imagesc([5.668 7.668],[7.425 4.425],e_img);
colormap(gray)
plot(xunit_e, yunit_e,'red');

imagesc([-13.67 -11.67],[-0.9996 -3.9996],f_img);
colormap(gray)
plot(xunit_f, yunit_f,'red');

imagesc([-8.849  -6.849],[-0.67263 -3.67263],g_img);
colormap(gray)
plot(xunit_g, yunit_g,'red');

imagesc([1.761  3.761],[-0.52332 -3.52332],i_img);
colormap(gray)
plot(xunit_i, yunit_i,'red');

imagesc([-1.001  -3.001],[-0.7126 -3.7126],h_img);
colormap(gray)
plot(xunit_h, yunit_h,'red');

imagesc([6.538  8.538],[-0.2954 -3.2953],j_img);
colormap(gray)
plot(xunit_j, yunit_j,'red');

imagesc([-7.446 -5.446],[-9.475 -12.475],k_img);
colormap(gray)
plot(xunit_k, yunit_k,'red');

imagesc([-2.5 -4.5],[-6.864 -9.864],l_img);
colormap(gray)
plot(xunit_l, yunit_l,'red');

imagesc([0.104 2.104],[-7.777 -10.777],m_img);
colormap(gray)
plot(xunit_m, yunit_m,'red');

imagesc([3.282 5.282],[-7.697 -10.697],n_img);
colormap(gray)
plot(xunit_n, yunit_n,'red');

imagesc([6.841 8.841],[-7.249 -10.249],o_img);
colormap(gray)
plot(xunit_o, yunit_o,'red');

imagesc([-3.69 -5.69],[-15.74 -18.74],p_img);
colormap(gray)
plot(xunit_p, yunit_p,'red');

imagesc([-1.4317 0.5683],[-13.68 -16.68],q_img);
colormap(gray)
plot(xunit_q, yunit_q,'red');

imagesc([3.29 5.29],[-13.61 -16.61],r_img);
colormap(gray)
plot(xunit_r, yunit_r,'red');


imagesc([9.07 11.07],[3.629 0.629],s_img);
colormap(gray)
plot(xunit_s, yunit_s,'red');

imagesc([12.64 14.64],[2.67 -0.33],t_img);
colormap(gray)
plot(xunit_t, yunit_t,'red');

imagesc([-11.55 -9.55],[10.7 7.7],z_img);
colormap(gray)
plot(xunit_z, yunit_z,'red');

imagesc([-3.926 -5.926],[12.12 9.12],z1_img);
colormap(gray)
plot(xunit_z1, yunit_z1,'red');