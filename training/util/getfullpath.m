function str=getfullpath(filename)
% Function: getfullpath
% Author:   Luca Cerone  
% Ver   :   1.0.0
%
% getfullpath reconstructs the full path to a folder or file starting from
% its relative path.
% It has been tested on Linux, but it should work on Windows and Mac as
% well.
%
% Syntax:
% fpath=getfullpath(relative_file_or_dir_name)
%
% Example 1:
%   (assuming to be in /home/user/Documents/Matlab)
%
%  fpath = getfullpath('../../Music/')
%     
%  fpath = 
%
%  /home/user/Music/
%
%  Example 2:
%     (assuming to be in /home/user/Documents/Matlab)
%
%  fpath=getfullpath('../../Music/Song.mp3')
%   
%  fpath =
%
%  /home/user/Music/Song.mp3

str=fliplr(pwd);

t='.';
prev_fold=strcmpi(t,'..');
curr_fold=strcmpi(t,'.');
while(prev_fold||curr_fold)
    [t,filename]=strtok(filename,filesep);
    prev_fold=strcmpi(t,'..');
    curr_fold=strcmpi(t,'.');
    if prev_fold
        [~,str]=strtok(str,filesep);
    end
    
    
end
str=[fliplr(str)];
if (not(str(end)==filesep))
    str=[str filesep];
end
str=[str t filename];

end