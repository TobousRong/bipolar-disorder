clc;clear;close all
%% Static analysis: This function calculates the calibrated Hin, Hse and HB from fMRI time series
Subj=textread('E:\Biplor\bip.txt','%s');
N=100;N_sub=98;
%mypool=parpool('local',24,'IdleTimeout',240);
Long_fmri=[];IN=[];IM=[];rIN={};rIM={};
parfor sub=50:98
    path=strcat('E:\Biplor\output\fMRI\',Subj(sub),'_Schaefer100_rest.mat');
    MRI=load(char(path));
    fmri=MRI.BOLD;
    Long_fmri=[Long_fmri;fmri];
    %% individual static FC matrix and its hierarchical module partition
    FC=corr(fmri);
    [Clus_num,Clus_size,mFC] = Functional_HP(FC,N);
    [Hin,Hse,R_Hin,R_Hse] =Seg_Int_component(FC,N,Clus_size,Clus_num);
    IN=[IN;Hin];IM=[IM;Hse];rIN{sub-49}=R_Hin;rIM{sub-49}=R_Hse;
end
%% Calibrating the individual static segregation and integration component
sFC=corr(Long_fmri);
[Hin,Hse] = Stable_correct(sFC,IN,IM,N);
rHin={};rHse={};
for sub=1:49
    rHin{sub}=rIN{sub}*Hin(sub)/mean(rIN{sub});
    rHse{sub}=rIM{sub}*Hse(sub)/mean(rIM{sub});
end
save('bip_static_rest.mat','Hin','Hse','rHin','rHse')
% HB=Hin-Hse

