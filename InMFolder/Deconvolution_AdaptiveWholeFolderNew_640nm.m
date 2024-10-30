close all;
clear all;


%%
RootFolder = 'D:Data\';
DataMatFolder = strcat(RootFolder, 'DataMatSaveFolder\');

%%
DeconvResultSaveFolder = strcat(RootFolder,'DeconvResultFolder','\');
DeconvResultImageSaveFolder = strcat(DeconvResultSaveFolder,'DeconvResultImageSaveFolder','\');
DeconvResultMatSaveFolder = strcat(DeconvResultSaveFolder,'DeconvResultMatSaveFolder','\');

if isfolder(DeconvResultSaveFolder) == 0
    mkdir(DeconvResultImageSaveFolder);
    mkdir(DeconvResultMatSaveFolder);
end



File = dir(fullfile(DataMatFolder,'*.mat'));
FileNames = {File.name}';
Number_File = length(FileNames);




I = load(strcat(DataMatFolder, FileNames{1}));
I = I.TestResult;
% I = I(1:600,1:700);

Size = size(I);
Size_x = Size(1,1);
Size_y = Size(1,2);

if mod(Size_x, 2) == 0
    Size_x_half = Size_x/2 + 1;
else 
    Size_x_half = (Size_x+1)/2;
end
if mod(Size_y, 2) == 0
    Size_y_half = Size_y/2 + 1;
else 
    Size_y_half = (Size_y+1)/2;
end



%% PSF
PSFObject_num = 38;
PSFBackground_num = 56;

% PSF_File = dir(fullfile(PSFRawFolder,'*.tif'));  
% PSF_FileNames = {PSF_File.name}';
% PSF_Number_File = length(PSF_FileNames);

Hologram_Psf = imread('D:Data\52.tif');
Hologram_Psf = double(Hologram_Psf);
Hologram_Psf_norm = (Hologram_Psf - min(min(Hologram_Psf))) ...
    ./(max(max(Hologram_Psf))-min(min(Hologram_Psf)));
%figure,imshow(Hologram_Psf_norm),title('Hologram of PSF'),colorbar

%psf = Hologram_Psf(71:450, 1:700);
psf = Hologram_Psf;
[max_x,max_y] = find(psf == max(max(psf)));
Size_psf = size(psf);
size_psf_x = Size_psf(1,1);
size_psf_y = Size_psf(1,2);
psf_central_x = max_x;
psf_central_y = max_y;

if Size_x_half-psf_central_x+1<1
    psf = psf(psf_central_x - Size_x_half +1:800, :);
end
if Size_y_half-psf_central_y+1<1
    psf = psf(:, round((size_psf_y - Size_y)/2+1):1100 - round((size_psf_y - Size_y)/2)-1-1);
end
[max_x,max_y] = find(psf == max(max(psf)));
psf_central_x = max_x;
psf_central_y = max_y;

Size_psf = size(psf);
size_psf_x = Size_psf(1,1);
size_psf_y = Size_psf(1,2);
psf_norm = (psf - min(min(psf)))./(max(max(psf)) - min(min(psf)));
% figure,imshow(psf_norm),title('Hologram of PSF'),colorbar

%psf_background = imread(strcat(PSFRawFolder, FileNames{PSFBackground_num}));
psf_background = imread('D:Data\dif_after_average0056_psf_background.tif');
psf_background = double(psf_background);
psf_back_size = size(psf_background);
psf_back_size_x = psf_back_size(1,1);
psf_back_size_y = psf_back_size(1,2);

if psf_back_size_x > Size_x
    psf_background = psf_background(1:Size_x, :);
else
    psf_background(psf_back_size_x+1: Size_x, :) = psf_background(1:Size_x-psf_back_size_x, :);
end
if psf_back_size_y > Size_y
    psf_background = psf_background(:, 1:Size_y);
else
    psf_background(:, psf_back_size_y+1: Size_y) = psf_background(:, 1:Size_y-psf_back_size_y);
end
    

psf_background_norm = (psf_background - min(min(psf_background))) ...
    ./(max(max(psf_background)) - min(min(psf_background)));
% figure,imshow(psf_background_norm)


psf_whole = psf_background*0.00315;
psf_whole(Size_x_half-psf_central_x+1:Size_x_half-psf_central_x+size_psf_x,...
    Size_y_half-psf_central_y+1:Size_y_half-psf_central_y+size_psf_y) = psf;

Hologram_Psf_norm = (psf_whole-min(min(psf_whole)))./(max(max(psf_whole))-min(min(psf_whole)));
%figure,imshow(Hologram_Psf_norm)



testInputsDir = strcat(RootFolder, 'DiffResultSaveFolder\DiffMat__UnNorm\');
InputFile = dir(fullfile(testInputsDir,'*mat.'));  
InputFileNames = {InputFile.name}';
Number_InputFile = length(InputFileNames);
kk = 1;
%% Particle
for num = 1:Number_InputFile
    %% Image Import
    if exist(strcat(DataMatFolder, num2str(num), '.mat'))==0 || exist(strcat(DeconvResultImageSaveFolder,num2str(num), '.tif'))~=0
         kk = kk + 1;
        
    else

        DiffRaw = load(strcat(DataMatFolder, num2str(num), '.mat'));
        DiffRaw = DiffRaw.TestResult;
        DiffRaw = double(DiffRaw);
        DiffRaw =  DiffRaw(1:Size_x, 1:Size_y);
        Hologram_Particles = DiffRaw;
        Hologram_Particles_norm = (Hologram_Particles - min(min(Hologram_Particles))) ...
            ./(max(max(Hologram_Particles)) - min(min(Hologram_Particles)));
        %figure,imshow(Hologram_Particles_norm),colorbar
        %figure,imshow(DiffRaw,[]), colorbar
        %% Parameter Simulation
        L= 640e-9;
        k0 = 2*pi/L;
        NA = 1.49;
        koff = k0*NA;
        thetai = 61.5;
        ps = 65e-9;
        dc3 = 1.33^2;
        % dc2 = -15.051+1.0516*1i;
        % dc2 = 2.0182
        dc2 = 20;
        dc2real = real(dc2);
        dc2imag = imag(dc2);
        d = 198e-9;
        dc1 = 1.515^2;
        thetap = 270;

        %% Calculation of Wave Vectors
        kspi = k0*sqrt(dc1)*sind(thetai); 
        kspr = k0*sqrt(dc2*dc3/(dc2+dc3)); % propagation constant
        kii = k0*power((real(dc2)*dc3)/(real(dc2)+dc3),3/2)*imag(dc2)/2/((real(dc2)).^2);

        %% Define Grids
        y = -Size_x/2:1:Size_x/2-1;
        x =  -Size_y/2:1:Size_y/2-1;
        X = ones(length(y),1)*x.*ps;
        Y = y'*ones(1,length(x)).*ps;
        KX1 = 2*pi/(ps*2)/(Size_y/2)*x;
        KY1 = 2*pi/(ps*2)/(Size_x/2)*y;
        KX = ones(length(KY1),1)*KX1;
        KY = KY1'*ones(1,length(KX1));
        r = sqrt(X.^2+Y.^2);
        Kr = sqrt(KX.^2+KY.^2);
        % figure,imshow(Kr,[])

        Fm_size = 20;
        xx = -Fm_size:1:Fm_size;
        yy = -Fm_size:1:Fm_size;
        KXX = ones(length(xx),1)*xx;
        KYY = yy'*ones(1,length(xx));
        Krr = sqrt(KXX.^2+KYY.^2);
        Krr_norm = Krr./max(max(Krr));
        Fm = ones(Size_x,Size_y);
        Fm(Size_x_half+1-Fm_size:Size_x_half+1+Fm_size,Size_y_half+1-Fm_size:Size_y_half+1+Fm_size) = Krr_norm;

        %% Reconstruction Algorithm
        Ei = exp(1i.*kspi.*(X.*cosd(thetap)+Y.*sind(thetap)));
        HzE = Hologram_Particles; 
        fm = exp(-abs(Kr-kspr).^2/(2*pi/(ps*2)/(Size_x/2)*5).^2).*(Kr<=kspr)+exp(-abs(Kr-kspr).^2/(2*pi/(ps*2)/(Size_x/2)*5).^2).*(Kr>kspr);
        % fm =exp(-abs(sqrt((KY-kspr*sind(thetap)).^2 +(KX-kspr*cosd(thetap)).^2) -kspr).^2/(2*pi/(ps)/220*2).^2).* ...
        %    ((((KY-kspr*sind(thetap)).^2 +(KX-kspr*cosd(thetap)).^2 -kspr.^2)<=0)) ...
        %    +exp(-abs(sqrt((KY-kspr*sind(thetap)).^2 +(KX-kspr*cosd(thetap)).^2) -kspr).^2/(2*pi/(ps*2)/220*5).^2).* ...
        %    ((((KY-kspr*sind(thetap)).^2 +(KX-kspr*cosd(thetap)).^2 -kspr.^2)>=0));
        % figure,imshow(abs(fm));
        HzE = ifft2(fftshift(fftshift(fft2(HzE)).*Fm));
        asHzE = fftshift(fft2(HzE.*Ei)); 
        asHzE(:,Size_y_half-1:Size_y_half+1) = 0;
        %figure,imshow(abs(asHzE),[])
        %figure,imshow(abs(fm),[])
        asHzE = asHzE.*fm;

        asHzE_norm = (asHzE - min(min(asHzE)))./(max(max(asHzE)) - min(min(asHzE)));
        % figure,imshow(abs(asHzE),[]);
        % 
        % figure,imshow(abs(fm),[]);
        % figure,imshow(abs((ifft2(asHzE.*fm))),[])

        %figure,imshow(abs(ifft2(asHzE)),[])

        psfHzE = Hologram_Psf_norm;
        psfHzE = ifft2(fftshift(fftshift(fft2(Hologram_Psf_norm)).*Fm));
        psfHzE = fftshift(fft2(psfHzE.*Ei)); 
        psfHzE(:,Size_y_half-1:Size_y_half-1+1) = 0;
        psfHzE = psfHzE.*fm;




        %figure,imshow(abs(ifft2(psfHzE)),[])

        % psfHzE_norm(:,321) = 0;

        %% Richardson–Lucy deconvolution
        ifft_asHzE = ifft2(fftshift(asHzE));


        ifft_psfHzE = ifft2(fftshift(psfHzE));  

        ifft_asHzE_norm = (ifft_asHzE - min(min(ifft_asHzE)))./(max(max(ifft_asHzE)) - min(min(ifft_asHzE)));
        ifft_psfHzE_norm = (ifft_psfHzE - min(min(ifft_psfHzE)))./(max(max(ifft_psfHzE)) - min(min(ifft_psfHzE)));


    %     luc1_abs = RL_Simple(abs(ifft_asHzE_norm./Ei),abs(ifft_psfHzE_norm./Ei),50);
    %     luc1_real_phase_simple = RL_Simple(ifft_asHzE_norm./Ei,ifft_psfHzE_norm./Ei,50);
    %     luc1_real_phase = Accelaration_DeconvLucy(ifft_asHzE_norm./Ei,ifft_psfHzE_norm./Ei,50,'Acce3');
    %     luc1_phase = Accelaration_DeconvLucy(angle(ifft_asHzE_norm),angle(ifft_psfHzE_norm),50);
    %     luc1_real_phase = Accelaration_DeconvLucy((ifft_asHzE_norm./Ei),(ifft_psfHzE_norm./Ei),50);
        %%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%
        luc1 = Accelaration_DeconvLucy(abs(ifft_asHzE./Ei),abs(ifft_psfHzE./Ei),50,'Acce3');
        %luc1 = Accelaration_DeconvLucy(angle(ifft_asHzE),angle(ifft_psfHzE),50,'Acce3');
        luc1_abs_norm = (luc1 - min(luc1, [], 'all'))./(max(luc1, [], 'all') - min(luc1, [], 'all'));
        luci_abs_uint16 = im2uint16(luc1_abs_norm);
    %     


        imwrite(luci_abs_uint16, strcat(DeconvResultImageSaveFolder,num2str(num), '.tif'));
        save(strcat(DeconvResultMatSaveFolder,num2str(num), '.mat'), 'luc1');
        %figure,imshow(luc1_abs_norm,[0.8,1]),colorbar;
        %figure,imagesc(luc1),colormap jet,colorbar;

        num
    end
  
end