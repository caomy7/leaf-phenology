

DirList.MOD15LAI='F:\01DBF\Input_pro\';
Stand=strcat([DirList.MOD15LAI,'MOD15A2H']);
MOD15=xlsread(Stand);
MOD15LAI=MOD15(:,1);

% process for other quality data (pick up)
index=find(MOD15(:,2)==1);
MOD15(index,1)=0;
MOD15MSK=MOD15(:,1);


% other quality data are filled using moving median value (replace the data that value=0)
MOD15MED = movmedian(MOD15LAI,2*3+1,'omitnan','Endpoints','shrink');
MOD15LAI(MOD15MSK==0) = MOD15MED(MOD15MSK==0);


    % fill gap
    MOD15GF = fillgaps(MOD15LAI,3);

    % remove spike
    MOD15HAM = hampel(MOD15GF);
    MOD15HAM = hampel(MOD15HAM);
    MOD15HAM = hampel(MOD15HAM);
    MOD15HAM = hampel(MOD15HAM);
    MOD15HAM = hampel(MOD15HAM);
    % savitzky-golay filter
    MOD15SG = sgolayfilt(MOD15HAM,3,7);

    MOD15LAI= reshape(MOD15SG,46,[]);  

    MOD15DLAI= interp1(4.5:8:365,MOD15LAI,1:365,'linear','extrap');
      MOD15DLAI(MOD15DLAI<0)=0;
      DLAI=MOD15DLAI;
      save('DLAI.mat','DLAI')

