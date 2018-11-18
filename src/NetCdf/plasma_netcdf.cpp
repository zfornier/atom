#include "../../include/NetCdf/plasma_netcdf.h"

namespace plasmanetcdf{

Error_t NetCDFManipulator::plsm_create(const char* fileName, const int mesh_size[3]){

    NcFile dataFile(fileName, NcFile::replace);
	NcDim xDim = dataFile.addDim("x", mesh_size[0]);
    NcDim yDim = dataFile.addDim("y", mesh_size[1]);
    NcDim zDim = dataFile.addDim("z", mesh_size[2]);

    return 0;
}


Error_t NetCDFManipulator::plsm_save_info(const char *fileName, const char* dim_names[], const int dim_sizes[], const int dim_nb) {
   NcFile dataFile(fileName, NcFile::write);

    // add dims to NetCDF file
    for (int i = 0; i< dim_nb; i++) {
        dataFile.addDim(dim_names[i], dim_sizes[i]);
    }

}

const char* NetCDFManipulator::plsm_get_description(const char* fileName, const char* variable){

	 NcFile dataFile(fileName, NcFile::read);

	NcVar var;
	NcVarAtt att;
	string str;

	var = dataFile.getVar(variable);

	if(var.isNull()) return NULL;
	  att = var.getAtt("description");
	if(att.isNull()) return NULL;

	att.getValues(str);
	
	char* desc = new char[str.size()+1];
		copy(str.begin(),str.end(),desc);
		desc[str.size()] = '\0';

	 
	return desc;
}




Error_t NetCDFManipulator::plsm_get_var(const char* fileName, const char * name, void* array) {
    NcFile dataFile(fileName, NcFile::read);
    NcVar var = dataFile.getVar(name);
    var.getVar(array);
    dataFile.close();
    return 0;

}

Error_t NetCDFManipulator::plsm_save_3D_double_array(const char* fileName, double* pTab, char* label,const char* unit,const char* desc) {   
    NcFile dataFile(fileName, NcFile::write);
    try
        {  
    
        NcDim xDim = dataFile.getDim("x");
        NcDim yDim = dataFile.getDim("y");
        NcDim zDim = dataFile.getDim("z");
    
        vector<NcDim> dims;
        dims.push_back(xDim);
        dims.push_back(yDim);
        dims.push_back(zDim);
        NcVar data = dataFile.addVar(label, ncDouble, dims);
    
        data.putVar(pTab);
        data.putAtt(UNITS,unit);
        data.putAtt(DESCRIPTION,desc);

        return 0;
    }
    catch(NcException& e)
    {
        cout << e.what();
        return PLSM_NC_ERROR;
    }

}

Error_t NetCDFManipulator::plsm_save_1D_int_array(const char* fileName, int* pTab, char* label, char* dim_label,const char* unit,const char* desc) {    
//cout << "write test" ;
    NcFile dataFile(fileName, NcFile::write);
    try
        {

        NcDim xDim = dataFile.getDim(dim_label);

        vector<NcDim> dims;
        dims.push_back(xDim);
        NcVar data = dataFile.addVar(label, ncDouble, dims);

        data.putVar(pTab);
        data.putAtt(UNITS,unit);
        data.putAtt(DESCRIPTION,desc);

        return 0;
    }
    catch(NcException& e)
    {
    cout << e.what();
      return PLSM_NC_ERROR;
    }
}

Error_t NetCDFManipulator::plsm_save_double(const char* fileName, double* pTab, char* label,const char* unit,const char* desc) {    
//cout << "write test" ;
    NcFile dataFile(fileName, NcFile::write);
    try
        {

        NcVar data = dataFile.addVar(label, ncDouble);

        data.putVar(pTab);
        data.putAtt(UNITS,unit);
        data.putAtt(DESCRIPTION,desc);

        return 0;
    }
    catch(NcException& e)
    {
    cout << e.what();
      return PLSM_NC_ERROR;
    }
}

Error_t NetCDFManipulator::plsm_save_int(const char* fileName, int* pTab, char* label,const char* unit,const char* desc) {
    //cout << "write test" ;
    NcFile dataFile(fileName, NcFile::write);
    try
        {

        NcVar data = dataFile.addVar(label, ncInt);

        data.putVar(pTab);
        data.putAtt(UNITS,unit);
        data.putAtt(DESCRIPTION,desc);

        return 0;
    }
    catch(NcException& e)
    {
    cout << e.what();
      return PLSM_NC_ERROR;
    }
}

Error_t NetCDFManipulator::plsm_add_dimension(const char *fname, const char* dim_name, int dim_size){
   NcFile dataFile(fname, NcFile::write);

    dataFile.addDim(dim_name, dim_size);

}


Error_t NetCDFManipulator::plsm_save_attribute(const char *fileName, const char* variable,const char* name,const char* desc){

	 NcFile dataFile(fileName,NcFile::write); 

	try
	    {
		NcVar var;
		var = dataFile.getVar(variable);
		if(var.isNull()) return PLSM_ERROR_FILE_NOT_FOUND;
		
		var.putAtt(name,desc);

		return 0;
	}
	 catch(NcException& e)
    	{
		cout << e.what();
		return PLSM_NC_ERROR;
    	}
	
}

}


