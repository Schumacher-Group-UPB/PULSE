clear;    %clear Matlab Workspace
rng('shuffle');
x0=10; y0=10;  width=1200; height=1000;
%------------------------------------------Misc------------------------------------------%
N = 500;                  %grid size
dxy = 0.4;                 %spatial discretization step
L = N*dxy;                 %real space grid length
x = sparse(dxy*(-N/2+1:N/2)).';   %x dimension discretization
y = sparse(dxy*(-N/2+1:N/2)).';   %y dimension discretization
kx = (2*pi*(-N/2+1:N/2)/L).';     %wavevector discretization
ky = (2*pi*(-N/2+1:N/2)/L).';     %wavevector discretization

%--------------------------------GP - parameter definition-------------------------------%
hbar = 6.582119569E-4;  %hbar in eVps^-1
mc = 5.684E-4;          %effective polariton mass
gammaC = 0.005;         %polariton loss rate in ps^-1
gc = 2E-6;              %polariton-polariton interaction strength in eVum^2
%--------------------------------GP - parameter definition-------------------------------%
for sec = 1:1%3
if  sec == 1
    wp = 4.5;
    xres=30; 
    yres=30;
    marray = [-1 1; 0 0; 1 -1; 0 1; 1 0; 1 1; 0 -1; -1 0; -1 -1];
elseif sec == 2
    wp = 8;
    xres=50; 
    yres=50;
    marray = [-3 1; -3 0; -3 -1; -1 3; 0 3; 1 3; -3 -1; -3 0; -3 1; 3 1; 3 0; 3 -1];
elseif sec == 3
    wp = 12;
    xres=70; 
    yres=70;
    marray = [0 -4; -4 0; 4 0; 0 4];
end

[rows,~] = size(marray);

for mscan = 1:1%rows

mp = marray(mscan,1);
mm = marray(mscan,2);
%-----------------------------------call-pulse-------------------------------------------%

str1 = 'main.exe -tetm --threads 4 ';                                   %call PULSE with TE-TM splitting and use 4 threads

str1 = 'phoenix_64.exe -tetm ';  
str2 = '--N 500 500 --L 100 100 --boundary zero zero ';                 %define the real-space grid
str3 = '--tmax 10000 ';                                                 %time settings
str4 = ['--initialState 0.1 add 70 70 0 0 plus 1 ',num2str(mp),' gauss+noDivide --initialState 0.1 add 70 70 0 0 minus 1 ',num2str(mm),' gauss+noDivide '];  %define initial condition
str5 = '--g_pm 6E-7 --deltaLT 0.025E-3 ';                               %set GP-Parameters
str6 = ['--pump 100 add ',num2str(wp),' ',num2str(wp),' 0 0 both 1 none gauss+noDivide+ring '];   %define pump
str7 = '--outEvery 5 --path data/results/';                                   %set output directory

inputstr = [str1,str2,str3,str4,str5,str6,str7];
[status,cmdout] = system(inputstr,'CUDA_VISIBLE_DEVICES','0');

htmlString = removeAnsiCodes(cmdout);
disp(htmlString(end-2630:end));

%----------------------------------post-processing---------------------------------------%
psiiniplus = readmatrix('data/results/initial_condition_plus.txt');
psiiniplus=psiiniplus(1:N,1:N)+1i*psiiniplus(N+1:2*N,1:N);
Y1iniplus = abs(reshape(psiiniplus,N,N)).^2;

psiiniminus = readmatrix('data/results/initial_condition_minus.txt');
psiiniminus=psiiniminus(1:N,1:N)+1i*psiiniminus(N+1:2*N,1:N);
Y1iniminus = abs(reshape(psiiniminus,N,N)).^2;

pump = readmatrix('data/results/pump_plus.txt');
pump=pump(1:N,1:N)+1i*pump(N+1:2*N,1:N);

psi = readmatrix('data/results/wavefunction_plus.txt');
psi=psi(1:N,1:N)+1i*psi(N+1:2*N,1:N);
Y1 = abs(reshape(psi,N,N)).^2;

psim = readmatrix('data/results/wavefunction_minus.txt');
psim=psim(1:N,1:N)+1i*psim(N+1:2*N,1:N);
Y1m = abs(reshape(psim,N,N)).^2;

figure(2);surf(x,y,Y1);shading interp;view(2);colormap('jet');pbaspect([1 1 1]);axis tight;set(gca,'XTickLabel',[]);set(gca,'YTickLabel',[]);shading interp;xlim([-xres xres]);ylim([-yres yres]);clim([0 160])
%saveas(gca,['Figure\msum=',num2str(mp+mm),'_mdif=',num2str(mp-mm),'_psip_d.png']);
figure(3);surf(x,y,Y1m);shading interp;view(2);colormap('jet');pbaspect([1 1 1]);axis tight;set(gca,'XTickLabel',[]);set(gca,'YTickLabel',[]);shading interp;xlim([-xres xres]);ylim([-yres yres]);clim([0 160])
%saveas(gca,['Figure\msum=',num2str(mp+mm),'_mdif=',num2str(mp-mm),'_psim_d.png']);
figure(4);surf(x,y,angle(reshape(psi,N,N)));shading interp;view(2);colormap('gray');pbaspect([1 1 1]);axis tight;set(gca,'XTickLabel',[]);set(gca,'YTickLabel',[]);shading interp;xlim([-xres xres]);ylim([-yres yres]);clim([-pi pi])
%saveas(gca,['Figure\msum=',num2str(mp+mm),'_mdif=',num2str(mp-mm),'_psip_arg.png']);
figure(5);surf(x,y,angle(reshape(psim,N,N)));shading interp;view(2);colormap('gray');pbaspect([1 1 1]);axis tight;set(gca,'XTickLabel',[]);set(gca,'YTickLabel',[]);shading interp;xlim([-xres xres]);ylim([-yres yres]);clim([-pi pi])
%saveas(gca,['Figure\msum=',num2str(mp+mm),'_mdif=',num2str(mp-mm),'_psim_arg.png']);

end
end
%----------------------------------String handeling functions----------------------------------%
function cleanedString = removeAnsiCodes(inputString)
    % Use regexprep to remove ANSI escape codes
    cleanedString = strrep(inputString, '[90m#[0m', ' ');
    cleanedString = regexprep(cleanedString, '\[0m|\[1m|\[2m|\[3m|\[4m|\[30m|\[31m|\[32m|\[33m|\[34m|\[35m|\[36m|\[37m|[93m|\[94m|\[?25h|\[2K|\[?25l|\[90m|\[A', '');
end