function T_fr = detect_trj(c_array,dc,dfr,Nfr_min)
%detect_trj links position data into trajectories
% T_fr = detect_trj(c_array,dc,dfr,Nfr_max)
%   INPUT:
%       c_array - cell array of detected particles. Each cell should represent a list of all particles in one frame
%       dc - maximum distance from the detected particle to look for linked
%            particles in other frames
%       dfr - number of frames to look for linked particles
%       Nfr_min -  minimum number of frames in trajectory
% 
%   OUTPUT:
%       T_fr - output table with columns: trj_num, frames, xy and trj_id
%  
% Author: Anatolii Kashchuk

T = cell2table(c_array,'VariableNames',"xy");
trj_num=size(c_array{1},1);
T.ids{1}=(1:trj_num)'; % set ids of all particles on the first frame
T.frames{1}=ones(trj_num,1);

for i=2:numel(c_array) % frames
    for j=1:size(c_array{i},1) % particles
        for k=1:dfr  
            % handle first frames
            if i-k<=0
                trj_num=trj_num+1;
                T.ids{i}(j,1)=trj_num;
                break
            end
            
            % find minimum distance and index of the closest particle
            [M,ind]=min(sqrt(sum( (c_array{i}(j,:)-c_array{i-k} ).^2,2)));
           
            if M<=dc
                % if within a specified distance - mark current particle as a part of trajectory
                T.ids{i}(j,1)=T.ids{i-k}(ind);
                break
            else
                % if all frames checked and no particle is close enough
                % create new trajectory id
                if k==dfr
                    trj_num=trj_num+1;
                    T.ids{i}(j,1)=trj_num;
                end
            end

        end
        
    end
    T.frames{i}=i*ones(j,1);
end

% trajectory ids for all detected points
trjids=sort(cat(1,T.ids{:}));

% number of frames for each trajectory
N_frames=diff(  find( diff([0;trjids;trj_num+1])~=0 )  ); % also adds numbers to include first and last trajectories into calculations

% Filter trajectories by frame number
trj_filt=find(N_frames>=Nfr_min);
T_ext=table;
T_ext.frames=cat(1,T.frames{:});
T_ext.xy=cat(1,T.xy{:});
T_ext.ids=cat(1,T.ids{:});


%% Output
T_fr=table('Size',[numel(trj_filt),4],'VariableTypes',{'int64','cell','cell','int64'},'VariableNames',{'trj_num','frames','xy','trj_id'});
for i=1:numel(trj_filt)
    Tnew=T_ext(T_ext.ids==trj_filt(i),:);
    T_fr(i,:).trj_num=i;
    T_fr(i,:).frames={Tnew.frames};
    T_fr(i,:).xy={Tnew.xy};
    T_fr(i,:).trj_id=Tnew(1,:).ids;
end


end