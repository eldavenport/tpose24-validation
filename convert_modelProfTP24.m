% get profiles from mitgcm output and compare to EQMix data
clear variables
addpath '/home/averdy/valid_matlab_needs/'

prof_data = '/data/SO3/edavenport/tpose24/profiles/'; % where the original profiles are (these are in several places, this begin one of them)
model_prof_data = '/data/SO3/edavenport/tpose24/oct2012_TP6Vel_3month/PROF/'; % where the model equivalent profiles are
output_netcdf_data = '/data/SO3/edavenport/tpose24/oct2012_TP6Vel_3month/PROF_eq/'; % where to put the converted profiles

list_model = {['TAO_WO_2012_ADCP_v2'],['TAO_WO_2012_CUR_v2'],['TAO_WO_2012_CTD_daily_ED'],['ADCP_prof_140'],['ADCP_prof_50'],['fastCTD_prof'],['FCTD_1min']};
MITprof_gcm2nc(prof_data,model_prof_data,output_netcdf_data,list_model);



