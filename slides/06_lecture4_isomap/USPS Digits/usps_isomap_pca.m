close all
load USPS_2digits
% Here we plot samples of our data set which is 
% USPS handwritten digits and we are using digit 2.

% This is how we extract digits 2.

%  idx=find (train_labels(3,:)==1);
%  o=train_patterns(:,idx);
%  xx=o(:,[1:109,111:296,298:475]);

faceW = 16; 
faceH = 16; 
numPerLine = 16; 
ShowLine = 8; 
Y = zeros(faceH*ShowLine,faceW*numPerLine); 
for i=0:ShowLine-1 
  	for j=0:numPerLine-1 
    	 Y(i*faceH+1:(i+1)*faceH,j*faceW+1:(j+1)*faceW) = reshape(xx(:,i*numPerLine+j+1),[faceH,faceW]); 
  	end 
end 
figure()
imagesc(Y');
colormap(gray);
axis off;
axis equal;
%% 


%% Isomap

% This is how we do the Isomap, we have commented the code and load the
% previously run version.The reason is that each time we run the code, the
% result may be different. As we are assigning images to the points in 2D
% (See the result) manually, we can't run from the begining.


%  m = size(xx, 2); 
% 
% G = sqrt(sum(xx.^2,1)'*ones(1,m) + ones(m,1)*sum(xx.^2,1) - 2*(xx'*xx));
%  e = 0.8*median(G(:));  
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
% zooo=find(D>10^20);
% %  D(D>10^20)=0;
%  
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
load 'dim1_usps'
load 'dim2_usps'

%%

% In this block we fit images of the digits to the points in 2D.

th = 0:pi/50:2*pi;
a=find(dim1>13.18 & dim1<13.2);
xunit_a = 0.5 * cos(th) + 13.19;
yunit_a = 0.5 * sin(th) + 8.177;

b=find(dim1>-10.02 & dim1<-10);
xunit_b = 0.5 * cos(th) + -10.01;
yunit_b = 0.5 * sin(th) + -5.913;

c=find(dim1>7.485 & dim1<7.487);
xunit_c = 0.5 * cos(th) + 7.486;
yunit_c = 0.5 * sin(th) + -7.533;

d=find(dim1>-10.34 & dim1<-10.32);
xunit_d = 0.5 * cos(th) + -10.33;
yunit_d = 0.5 * sin(th) + 5.519;

e=find(dim1>-5.193 & dim1<-5.191);
xunit_e = 0.5 * cos(th) + -5.192;
yunit_e = 0.5 * sin(th) + 5.242;

f=find(dim1>-16.67 & dim1<-16.65);
xunit_f = 0.5 * cos(th) + -16.66;
yunit_f = 0.5 * sin(th) + 4.828;

g=find(dim1>-10.05 & dim1<-10.03);
xunit_g = 0.5 * cos(th) + -10.04;
yunit_g = 0.5 * sin(th) + 0.2263;

h=find(dim1>-1.16 & dim1<-1.14);
xunit_h = 0.5 * cos(th) + -1.15;
yunit_h = 0.5 * sin(th) + 5.008;

i=find(dim1>3.74 & dim1<3.75);
xunit_i = 0.5 * cos(th) + 3.745;
yunit_i = 0.5 * sin(th) + 5.016;

j=find(dim1>8.25 & dim1<8.26);
xunit_j = 0.5 * cos(th) + 8.256;
yunit_j = 0.5 * sin(th) + 5.347;

k=find(dim1>6.09 & dim1<6.1);
xunit_k = 0.5 * cos(th) + 6.095;
yunit_k = 0.5 * sin(th) + 9.156;

l=find(dim1>1.03 & dim1<1.035);
xunit_l = 0.5 * cos(th) + 1.032;
yunit_l = 0.5 * sin(th) + 9.053;

n=find(dim1>-2.438 & dim1<-2.436);
xunit_n = 0.5 * cos(th) + -2.437;
yunit_n = 0.5 * sin(th) + 9.062;

o=find(dim1>6.585 & dim1<6.59);
xunit_o = 0.5 * cos(th) + 6.589;
yunit_o = 0.5 * sin(th) + 11.77;

p=find(dim1>-5.175 & dim1<-5.17);
xunit_p = 0.5 * cos(th) + -5.171;
yunit_p = 0.5 * sin(th) + 0.39;

q=find(dim1>-0.6052 & dim1<-0.605);
xunit_q = 0.5 * cos(th) + -0.6051;
yunit_q = 0.5 * sin(th) + 0.1347;

r=find(dim1>4.46 & dim1<4.47);
xunit_r = 0.5 * cos(th) + 4.465;
yunit_r = 0.5 * sin(th) + -0.002556;

s=find(dim1>9.07 & dim1<9.09);
xunit_s = 0.5 * cos(th) + 9.08;
yunit_s = 0.5 * sin(th) + 0.5817;

t=find(dim1>-6.015 & dim1<-6.01);
xunit_t = 0.5 * cos(th) + -6.013;
yunit_t = 0.5 * sin(th) + -5.145;

u=find(dim1>-1.06 & dim1<-1.058);
xunit_u = 0.5 * cos(th) + -1.059;
yunit_u = 0.5 * sin(th) + -4.821;

v=find(dim1>3.94 & dim1<3.942);
xunit_v = 0.5 * cos(th) + 3.941;
yunit_v = 0.5 * sin(th) + -4.683;

w=find(dim1>0.01 & dim1<0.015);
xunit_w = 0.5 * cos(th) + 0.01467;
yunit_w = 0.5 * sin(th) + -9.82;

z=find(dim1>-6.265 & dim1<-6.26);
xunit_z = 0.5 * cos(th) + -6.262;
yunit_z = 0.5 * sin(th) + -9.524;

z1=find(dim1>-15.05 & dim1<-14.95);
xunit_z1 = 0.5 * cos(th) + -15;
yunit_z1 = 0.5 * sin(th) + -4.335;

z2=find(dim1>10.87 & dim1<10.92);
xunit_z2 = 0.5 * cos(th) + 10.9;
yunit_z2 = 0.5 * sin(th) + -5.869;

z3=find(dim1>-13.75 & dim1<-13.7);
xunit_z3 = 0.5 * cos(th) + -13.72;
yunit_z3 = 0.5 * sin(th) + 1.335;

z4=find(dim1>11.55 & dim1<11.65);
xunit_z4 = 0.5 * cos(th) + 11.6;
yunit_z4 = 0.5 * sin(th) + 2.763;

z5=find(dim1>15.53 & dim1<15.55);
xunit_z5 = 0.5 * cos(th) + 15.54;
yunit_z5 = 0.5 * sin(th) + -11.3;

a_img=reshape(xx(:,135),[16 16]);
b_img=reshape(xx(:,328),[16 16]);

c_img=reshape(xx(:,410),[16 16]);
d_img=reshape(xx(:,366),[16 16]);
e_img=reshape(xx(:,171),[16 16]);
f_img=reshape(xx(:,71),[16 16]);
g_img=reshape(xx(:,129),[16 16]);
h_img=reshape(xx(:,298),[16 16]);
i_img=reshape(xx(:,327),[16 16]);
j_img=reshape(xx(:,215),[16 16]);
k_img=reshape(xx(:,294),[16 16]);

l_img=reshape(xx(:,338),[16 16]);
n_img=reshape(xx(:,210),[16 16]);
o_img=reshape(xx(:,399),[16 16]);
p_img=reshape(xx(:,227),[16 16]);
q_img=reshape(xx(:,284),[16 16]);
r_img=reshape(xx(:,176),[16 16]);
s_img=reshape(xx(:,361),[16 16]);
t_img=reshape(xx(:,288),[16 16]);
u_img=reshape(xx(:,341),[16 16]);
v_img=reshape(xx(:,222),[16 16]);
w_img=reshape(xx(:,375),[16 16]);
z_img=reshape(xx(:,110),[16 16]);
z1_img=reshape(xx(:,257),[16 16]);
z2_img=reshape(xx(:,14),[16 16]);
z3_img=reshape(xx(:,85),[16 16]);
z4_img=reshape(xx(:,382),[16 16]);
z5_img=reshape(xx(:,152),[16 16]);


figure()

scatter(dim1, dim2,18*ones(473,1),'fill');
title('Isomap');
hold on
imagesc([12.19 14.19],[7.577 5.577],a_img');
colormap(gray)
plot(xunit_a, yunit_a,'red');

imagesc([-11.01 -9.01],[-6.513 -8.513],b_img');
colormap(gray)
plot(xunit_b, yunit_b,'red');

imagesc([6.486 8.486],[-8.133 -10.133],c_img');
colormap(gray)
plot(xunit_c, yunit_c,'red');

imagesc([-11.33 -9.33],[4.919 2.919],d_img');
colormap(gray)
plot(xunit_d, yunit_d,'red');

imagesc([-6.192 -4.192],[4.642 2.642],e_img');
colormap(gray)
plot(xunit_e, yunit_e,'red');

imagesc([-17.66 -15.66],[4.228 2.228],f_img');
colormap(gray)
plot(xunit_f, yunit_f,'red');

imagesc([-11.04 -9.04],[-0.5837 -2.5873],g_img');
colormap(gray)
plot(xunit_g, yunit_g,'red');

imagesc([-2.15 -0.15],[4.408 2.408],h_img');
colormap(gray)
plot(xunit_h, yunit_h,'red');

imagesc([2.745 4.745],[4.016 2.016],i_img');
colormap(gray)
plot(xunit_i, yunit_i,'red');

imagesc([7.256 9.256],[4.747 2.747],j_img');
colormap(gray)
plot(xunit_j, yunit_j,'red');


imagesc([5.095 7.095],[8.556 6.556],k_img');
colormap(gray)
plot(xunit_k, yunit_k,'red');

imagesc([0.032 2.032],[8.453 6.453],l_img');
colormap(gray)
plot(xunit_l, yunit_l,'red');

imagesc([-1.437 -3.437],[8.462 6.462],n_img');
colormap(gray)
plot(xunit_n, yunit_n,'red');

imagesc([7.189 9.189],[12.77 10.77],o_img');
colormap(gray)
plot(xunit_o, yunit_o,'red');

imagesc([-6.171 -4.171],[-0.21 -2.21],p_img');
colormap(gray)
plot(xunit_p, yunit_p,'red');

imagesc([-1.6051 0.3949],[-0.4653 -2.4653],q_img');
colormap(gray)
plot(xunit_q, yunit_q,'red');

imagesc([3.465 5.465],[-0.597444 -2.597444],r_img');
colormap(gray)
plot(xunit_r, yunit_r,'red');

imagesc([8.08 10.08],[-0.0183 -2.0183],s_img');
colormap(gray)
plot(xunit_s, yunit_s,'red');

imagesc([-7.013 -5.013],[-5.745 -7.745],t_img');
colormap(gray)
plot(xunit_t, yunit_t,'red');

imagesc([-2.059 -0.059],[-5.421 -7.421],u_img');
colormap(gray)
plot(xunit_u, yunit_u,'red');

imagesc([2.941 4.491],[-5.283 -7.283],v_img');
colormap(gray)
plot(xunit_v, yunit_v,'red');

imagesc([-0.98533 1.01467],[-10.42 -12.42],w_img');
colormap(gray)
plot(xunit_w, yunit_w,'red');

imagesc([-7.262 -5.262],[-10.124 -12.124],z_img');
colormap(gray)
plot(xunit_z, yunit_z,'red');

imagesc([-16 -14],[-4.995 -6.995],z1_img');
colormap(gray)
plot(xunit_z1, yunit_z1,'red');

imagesc([9.9 11.9],[-6.469 -8.469],z2_img');
colormap(gray)
plot(xunit_z2, yunit_z2,'red');

imagesc([-14.72 -12.72],[0.335 -1.665],z3_img');
colormap(gray)
plot(xunit_z3, yunit_z3,'red');

imagesc([10.6 12.6],[2.163 0.163],z4_img');
colormap(gray)
plot(xunit_z4, yunit_z4,'red');


imagesc([14.54 16.54],[-11.9 -13.9],z5_img');
colormap(gray)
plot(xunit_z5, yunit_z5,'red');

 
 %% PCA
 % This is how we do PCA. Similarly, we have commented the code below and
 % we load a previously run version of this PCA.
 
 
%  center=mean(xx,2);
%  sz=size(xx,2);
%  x = xx - center * ones(1, sz);
%  covariance = cov(x'); 
% % [U, S] = eig(covariance); 
% [V, S] = eigs(covariance, 3); 
% 
% % proect x to the principal direction; 
% Vx = V(:,1:2)' * x; 

%%
load 'Vx_USPS'

%%
th = 0:pi/50:2*pi;
a=find(Vx(1,:)>-6.78 & Vx(1,:)<-6.77);
xunit_a = 0.25 * cos(th) + -6.771;
yunit_a = 0.25 * sin(th) + 3.463;

b=find(Vx(1,:)>-4.482 & Vx(1,:)<-4.48);
xunit_b = 0.25 * cos(th) + -4.481;
yunit_b = 0.25 * sin(th) + 3.729;

c=find(Vx(1,:)>-2.495 & Vx(1,:)<-2.493);
xunit_c = 0.25 * cos(th) + -2.494;
yunit_c = 0.25 * sin(th) + 4.434;

d=find(Vx(1,:)>0.117 & Vx(1,:)<0.118);
xunit_d = 0.25 * cos(th) + 0.1177;
yunit_d = 0.25 * sin(th) + 4.298;

e=find(Vx(1,:)>2.308 & Vx(1,:)<2.31);
xunit_e = 0.25 * cos(th) + 2.309;
yunit_e = 0.25 * sin(th) + 4.504;

f=find(Vx(1,:)>4.394 & Vx(1,:)<4.396);
xunit_f = 0.25 * cos(th) + 4.395;
yunit_f = 0.25 * sin(th) + 4.905;

g=find(Vx(1,:)>-6.728 & Vx(1,:)<-6.726);
xunit_g = 0.25 * cos(th) + -6.727;
yunit_g = 0.25 * sin(th) + 1.272;

h=find(Vx(1,:)>-4.346 & Vx(1,:)<-4.344);
xunit_h = 0.25 * cos(th) + -4.345;
yunit_h = 0.25 * sin(th) + 1.48;

i=find(Vx(1,:)>-2.248 & Vx(1,:)<-2.246);
xunit_i = 0.25 * cos(th) + -2.247;
yunit_i = 0.25 * sin(th) + 1.581;

j=find(Vx(1,:)>-0.4443 & Vx(1,:)<-0.4441);
xunit_j = 0.25 * cos(th) + -0.4442;
yunit_j = 0.25 * sin(th) + 1.485;

k=find(Vx(1,:)>1.821 & Vx(1,:)<1.823);
xunit_k = 0.25 * cos(th) + 1.822;
yunit_k = 0.25 * sin(th) + 1.647;

l=find(Vx(1,:)>4.12 & Vx(1,:)<4.122);
xunit_l = 0.25 * cos(th) + 4.121;
yunit_l = 0.25 * sin(th) + 1.935;

m=find(Vx(1,:)>6.377 & Vx(1,:)<6.379);
xunit_m = 0.25 * cos(th) + 6.378;
yunit_m = 0.25 * sin(th) + 2.077;

n=find(Vx(1,:)>-7.33 & Vx(1,:)<-7.31);
xunit_n = 0.25 * cos(th) + -7.32;
yunit_n = 0.25 * sin(th) + -1.07;

o=find(Vx(1,:)>-4.92 & Vx(1,:)<-4.9);
xunit_o = 0.25 * cos(th) + -4.91;
yunit_o = 0.25 * sin(th) + -1.036;

p=find(Vx(1,:)>-2.898 & Vx(1,:)<-2.896);
xunit_p = 0.25 * cos(th) + -2.897;
yunit_p = 0.25 * sin(th) + -0.7345;

q=find(Vx(1,:)>-0.6759 & Vx(1,:)<-0.6757);
xunit_q = 0.25 * cos(th) + -0.6758;
yunit_q = 0.25 * sin(th) + -0.9005;

r=find(Vx(1,:)>1.39 & Vx(1,:)<1.393);
xunit_r = 0.25 * cos(th) + 1.392;
yunit_r = 0.25 * sin(th) + -0.8453;

s=find(Vx(1,:)>3.65 & Vx(1,:)<3.67);
xunit_s = 0.25 * cos(th) + 3.66;
yunit_s = 0.25 * sin(th) + -0.563;

t=find(Vx(1,:)>5.91 & Vx(1,:)<5.92);
xunit_t = 0.25 * cos(th) + 5.919;
yunit_t = 0.25 * sin(th) + -0.2616;

u=find(Vx(1,:)>-2.74 & Vx(1,:)<-2.72);
xunit_u = 0.25 * cos(th) + -2.73;
yunit_u = 0.25 * sin(th) + -2.903;

w=find(Vx(1,:)>-0.579 & Vx(1,:)<-0.577);
xunit_w = 0.25 * cos(th) + -0.578;
yunit_w = 0.25 * sin(th) + -2.922;

z=find(Vx(1,:)>1.198 & Vx(1,:)<1.2);
xunit_z = 0.25 * cos(th) + 1.199;
yunit_z = 0.25 * sin(th) + -3.687;

z1=find(Vx(1,:)>2.69 & Vx(1,:)<2.71);
xunit_z1 = 0.25 * cos(th) + 2.7;
yunit_z1 = 0.25 * sin(th) + -3.016;

z2=find(Vx(1,:)>-1.795 & Vx(1,:)<-1.793);
xunit_z2 = 0.25 * cos(th) + -1.794;
yunit_z2 = 0.25 * sin(th) + -5.453;

z3=find(Vx(1,:)>-1.254 & Vx(1,:)<-1.252);
xunit_z3 = 0.25 * cos(th) + -1.253;
yunit_z3 = 0.25 * sin(th) + 5.489;

z4=find(Vx(1,:)>2.247 & Vx(1,:)<2.249);
xunit_z4 = 0.25 * cos(th) + 2.248;
yunit_z4 = 0.25 * sin(th) + 6.479;

z5=find(Vx(1,:)>4.496 & Vx(1,:)<4.498);
xunit_z5 = 0.25 * cos(th) + 4.497;
yunit_z5 = 0.25 * sin(th) + -2.404;

a_img=reshape(xx(:,119),[16 16]);
b_img=reshape(xx(:,423),[16 16]);
c_img=reshape(xx(:,415),[16 16]);
d_img=reshape(xx(:,72),[16 16]);
e_img=reshape(xx(:,121),[16 16]);
f_img=reshape(xx(:,173),[16 16]);
g_img=reshape(xx(:,418),[16 16]);
h_img=reshape(xx(:,18),[16 16]);
i_img=reshape(xx(:,67),[16 16]);
j_img=reshape(xx(:,387),[16 16]);
k_img=reshape(xx(:,358),[16 16]);
l_img=reshape(xx(:,35),[16 16]);
m_img=reshape(xx(:,286),[16 16]);
n_img=reshape(xx(:,147),[16 16]);
o_img=reshape(xx(:,165),[16 16]);
p_img=reshape(xx(:,444),[16 16]);
q_img=reshape(xx(:,70),[16 16]);
r_img=reshape(xx(:,64),[16 16]);
s_img=reshape(xx(:,425),[16 16]);

t_img=reshape(xx(:,152),[16 16]);
u_img=reshape(xx(:,34),[16 16]);

w_img=reshape(xx(:,317),[16 16]);
z_img=reshape(xx(:,211),[16 16]);
z1_img=reshape(xx(:,208),[16 16]);
z2_img=reshape(xx(:,182),[16 16]);
z3_img=reshape(xx(:,2),[16 16]);
z4_img=reshape(xx(:,379),[16 16]);
z5_img=reshape(xx(:,14),[16 16]);

figure()
scatter(Vx(1,:), Vx(2,:),18*ones(473,1),'fill');
title('PCA');
hold on
imagesc([-7.171 -6.371],[3.163 2.363],a_img');
colormap(gray)
plot(xunit_a, yunit_a,'red');

imagesc([-4.881 -4.081],[3.429 2.629],b_img');
colormap(gray)
plot(xunit_b, yunit_b,'red');

imagesc([-2.894 -2.094],[4.134 3.334],c_img');
colormap(gray)
plot(xunit_c, yunit_c,'red');

imagesc([-0.2823 0.5177],[3.998 3.198],d_img');
colormap(gray)
plot(xunit_d, yunit_d,'red');

imagesc([-0.2823 0.5177],[3.998 3.198],d_img');
colormap(gray)
plot(xunit_d, yunit_d,'red');

imagesc([1.909 2.709],[4.204 3.404],e_img');
colormap(gray)
plot(xunit_e, yunit_e,'red');

imagesc([3.995 4.795],[4.605 3.805],f_img');
colormap(gray)
plot(xunit_f, yunit_f,'red');

imagesc([-7.127 -6.327],[0.972 0.172],g_img');
colormap(gray)
plot(xunit_g, yunit_g,'red');

imagesc([-4.745 -3.945],[1.18 0.38],h_img');
colormap(gray)
plot(xunit_h, yunit_h,'red');

imagesc([-2.647 -1.847],[1.281 0.481],i_img');
colormap(gray)
plot(xunit_i, yunit_i,'red');

imagesc([-0.8442 0.0442],[1.185 0.385],j_img');
colormap(gray)
plot(xunit_j, yunit_j,'red');

imagesc([1.422 2.222],[1.347 0.547],k_img');
colormap(gray)
plot(xunit_k, yunit_k,'red');

imagesc([3.721 4.521],[1.635 0.835],l_img');
colormap(gray)
plot(xunit_l, yunit_l,'red');

imagesc([3.721 4.521],[1.635 0.835],l_img');
colormap(gray)
plot(xunit_l, yunit_l,'red');

imagesc([5.978 6.778],[1.777 0.977],m_img');
colormap(gray)
plot(xunit_m, yunit_m,'red');

imagesc([-7.72 -6.92],[-1.47 -2.27],n_img');
colormap(gray)
plot(xunit_n, yunit_n,'red');

imagesc([-5.31 -4.51],[-1.336 -2.136],o_img');
colormap(gray)
plot(xunit_o, yunit_o,'red');

imagesc([-3.297 -2.497],[-1.0345 -1.8345],p_img');
colormap(gray)
plot(xunit_p, yunit_p,'red');

imagesc([-1.0758 -0.2758],[-1.2005 -2.005],q_img');
colormap(gray)
plot(xunit_q, yunit_q,'red');

imagesc([0.992 1.792],[-1.1453 -1.9453],r_img');
colormap(gray)
plot(xunit_r, yunit_r,'red');

imagesc([3.26 4.06],[-0.863 -1.663],s_img');
colormap(gray)
plot(xunit_s, yunit_s,'red');

imagesc([5.519 6.319],[-0.5616 -1.3616],t_img');
colormap(gray)
plot(xunit_t, yunit_t,'red');

imagesc([-3.13 -2.33],[-3.203 -4.003],u_img');
colormap(gray)
plot(xunit_u, yunit_u,'red');

imagesc([-0.978 -0.178],[-3.222 -4.022],w_img');
colormap(gray)
plot(xunit_w, yunit_w,'red');

imagesc([0.799 1.599],[-3.987 -4.787],z_img');
colormap(gray)
plot(xunit_z, yunit_z,'red');

imagesc([2.3 3.1],[-3.316 -4.116],z1_img');
colormap(gray)
plot(xunit_z1, yunit_z1,'red');

imagesc([-2.894 -2.094],[-4.653 -5.453],z2_img');
colormap(gray)
plot(xunit_z2, yunit_z2,'red');

imagesc([-1.653 -0.853],[5.189 4.389],z3_img');
colormap(gray)
plot(xunit_z3, yunit_z3,'red');

imagesc([1.848 1.048],[6.179 5.379],z4_img');
colormap(gray)
plot(xunit_z4, yunit_z4,'red');

imagesc([4.097 4.897],[-2.704 -3.504],z5_img');
colormap(gray)
plot(xunit_z5, yunit_z5,'red');