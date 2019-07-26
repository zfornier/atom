#include "../../../include/NetCdf/plasma_netcdf.h"
#include <string.h>

#define CHK_NCERR(err) { \
     if((err) != NC_NOERR) { \
         printf("Error %d, %s\n in %s:%d\n", err, nc_strerror(err), __FILE__, __LINE__); \
         exit(1); \
     } \
     }

namespace plasmanetcdf {

    Error_t NetCDFManipulator::plsm_create(const char *fileName, const int mesh_size[3]) {
       /* NcFile dataFile(fileName, NcFile::replace);
        NcDim xDim = dataFile.addDim("x", mesh_size[0]);
        NcDim yDim = dataFile.addDim("y", mesh_size[1]);
        NcDim zDim = dataFile.addDim("z", mesh_size[2]);
        */

        int ncid;
        int dimId[3];

        CHK_NCERR(nc_create(fileName, NC_NETCDF4|NC_CLOBBER, &ncid));
        CHK_NCERR(nc_def_dim(ncid, "Nx", mesh_size[0], &dimId[0]));
        CHK_NCERR(nc_def_dim(ncid, "Ny", mesh_size[1], &dimId[1]));
        CHK_NCERR(nc_def_dim(ncid, "Nz", mesh_size[2], &dimId[2]));
        CHK_NCERR(nc_close(ncid));

        return 0;
    }


    Error_t NetCDFManipulator::plsm_save_info(const char *fileName, const char *dim_names[], const int dim_sizes[], const int dim_nb) {
        /*NcFile dataFile(fileName, NcFile::write);

        // add dims to NetCDF file
        for (int i = 0; i < dim_nb; i++) {
            dataFile.addDim(dim_names[i], dim_sizes[i]);
        }
	*/

	int ncid;
        int dimId[dim_nb];

        CHK_NCERR(nc_open(fileName,NC_WRITE,&ncid));

        for (int i = 0; i < dim_nb; i++) {
          CHK_NCERR(nc_def_dim(ncid,dim_names[i],dim_sizes[i],&dimId[i]));
        }

        CHK_NCERR(nc_close(ncid));

        return 0;
    }

    const char *NetCDFManipulator::plsm_get_description(const char *fileName, const char *variable) {
	
        /*NcFile dataFile(fileName, NcFile::read);

        NcVar var;
        NcVarAtt att;
        string str;

        var = dataFile.getVar(variable);

        if (var.isNull()) return NULL;
        att = var.getAtt("description");
        if (att.isNull()) return NULL;

        att.getValues(str);

        char *desc = new char[str.size() + 1];
        copy(str.begin(), str.end(), desc);
        desc[str.size()] = '\0';
	*/
	
	
  	int ncid;
        CHK_NCERR(nc_open(fileName,NC_WRITE,&ncid));

        int varid;
        CHK_NCERR(nc_inq_varid(ncid,variable,&varid));

        char var;
        CHK_NCERR(nc_get_var_text(ncid,varid,&var));
        if (var==0) return 0;

	char* att[250];

        CHK_NCERR(nc_get_att_string(ncid,varid,"Description",att));
        if (att==0) return 0;

	
        char *desc = new char[251];
        copy(att[0], att[249], desc);
        desc[250] = '\0';

	CHK_NCERR(nc_close(ncid));

        return desc;

    }


    Error_t NetCDFManipulator::plsm_get_var(const char *fileName, const char *name, void *array) {

        /*NcFile dataFile(fileName, NcFile::read);
        NcVar var = dataFile.getVar(name);
        var.getVar(array);
        dataFile.close();
	*/

	int ncid;
        CHK_NCERR(nc_open(fileName,NC_NOWRITE,&ncid));

	int varid;
        CHK_NCERR(nc_inq_varid(ncid,name,&varid));


        CHK_NCERR(nc_get_var(ncid,varid,array));

	CHK_NCERR(nc_close(ncid))

        return 0;
    }



    Error_t NetCDFManipulator::plsm_get_dim_var(const char *fileName, const char *name, int *dimVar) {

        /*NcFile dataFile(fileName, NcFile::read);
        NcDim dim = dataFile.getDim(name);
        *dimVar = (int)dim.getSize();
        dataFile.close();
	*/

	int ncid;
        CHK_NCERR(nc_open(fileName,NC_WRITE,&ncid));

	int idp;
	CHK_NCERR(nc_inq_dimid(ncid,name,&idp));

	size_t size;
	
	CHK_NCERR(nc_inq_dimlen(ncid,idp,&size));

	*dimVar=(int)size;

	CHK_NCERR(nc_close(ncid));


        return 0;
    }

    Error_t NetCDFManipulator::plsm_save_3D_double_array(const char *fileName, double *pTab, char *label, const char *unit, const char *desc) {
        
	/*NcFile dataFile(fileName, NcFile::write);
        try {
            NcDim xDim = dataFile.getDim("x");
            NcDim yDim = dataFile.getDim("y");
            NcDim zDim = dataFile.getDim("z");

            vector <NcDim> dims;
            dims.push_back(xDim);
            dims.push_back(yDim);
            dims.push_back(zDim);
            NcVar data = dataFile.addVar(label, ncDouble, dims);

            data.putVar(pTab);
            data.putAtt(UNITS, unit);
            data.putAtt(DESCRIPTION, desc);

            return 0;
        }
        catch (NcException &e) {
            cout << e.what();
            return PLSM_NC_ERROR;
        }
	*/

	int ncid;
        CHK_NCERR(nc_open(fileName,NC_WRITE,&ncid));


	int dims[3];
	CHK_NCERR(nc_inq_dimid(ncid,"x",&dims[0]));		
	CHK_NCERR(nc_inq_dimid(ncid,"y",&dims[1]));
	CHK_NCERR(nc_inq_dimid(ncid,"z",&dims[2]));

	int varid;
	
	CHK_NCERR(nc_def_var(ncid,label,NC_DOUBLE,3,dims,&varid));

	CHK_NCERR(nc_put_var_double(ncid,varid,pTab));

	CHK_NCERR(nc_put_att_text(ncid,varid,DESCRIPTION.c_str(),1,desc));
	CHK_NCERR(nc_put_att_text(ncid,varid,UNITS.c_str(),1,unit));
		
	CHK_NCERR(nc_close(ncid));

	return 0;


    }

    Error_t NetCDFManipulator::plsm_save_1D_double_array(const char *fileName, double *pTab, const char *label, const char *dim_label, const char *unit, const char *desc) {
        
	/*NcFile dataFile(fileName, NcFile::write);
        try {
            NcDim xDim = dataFile.getDim(dim_label);

            vector <NcDim> dims;
            dims.push_back(xDim);
            NcVar data = dataFile.addVar(label, ncDouble, dims);

            data.putVar(pTab);
            data.putAtt(UNITS, unit);
            data.putAtt(DESCRIPTION, desc);

            return 0;
        }
        catch (NcException &e) {
            cout << e.what();
            return PLSM_NC_ERROR;
        }

	*/

	int ncid;
        CHK_NCERR(nc_open(fileName,NC_WRITE,&ncid));

	int dimid[1];
	CHK_NCERR(nc_inq_dimid(ncid,dim_label,&dimid[0]));

	int varid;
	
	CHK_NCERR(nc_def_var(ncid,label,NC_DOUBLE,1,dimid,&varid));

	CHK_NCERR(nc_put_var_double(ncid,varid,pTab));

	CHK_NCERR(nc_put_att_text(ncid,varid,DESCRIPTION.c_str(),strlen(desc),desc));
	CHK_NCERR(nc_put_att_text(ncid,varid,UNITS.c_str(),1,unit));
		
	CHK_NCERR(nc_close(ncid));
	
	return 0;
	
    }

    Error_t NetCDFManipulator::plsm_save_double(const char *fileName, double *pTab, const char *label, const char *unit, const char *desc) {


        /*NcFile dataFile(fileName, NcFile::write);
        try {
            NcVar data = dataFile.addVar(label, ncDouble);

            data.putVar(pTab);
            data.putAtt(UNITS, unit);
            data.putAtt(DESCRIPTION, desc);

            return 0;
        }
        catch (NcException &e) {
            cout << e.what();
            return PLSM_NC_ERROR;
        }
	*/

	int ncid;
        CHK_NCERR(nc_open(fileName,NC_WRITE,&ncid));

	int varid;
	
	CHK_NCERR(nc_def_var(ncid,label,NC_DOUBLE,0,NULL,&varid));
	
	CHK_NCERR(nc_put_var_double(ncid,varid,pTab));

	CHK_NCERR(nc_put_att_text(ncid,varid,DESCRIPTION.c_str(),1,desc));
	CHK_NCERR(nc_put_att_text(ncid,varid,UNITS.c_str(),1,unit));
		
	CHK_NCERR(nc_close(ncid));
	
	return 0;



	
    }

    Error_t NetCDFManipulator::plsm_save_int(const char *fileName, int *pTab, const char *label, const char *unit, const char *desc) {
        /*NcFile dataFile(fileName, NcFile::write);
        try {

            NcVar data = dataFile.addVar(label, ncInt);

            data.putVar(pTab);
            data.putAtt(UNITS, unit);
            data.putAtt(DESCRIPTION, desc);

            return 0;
        }
        catch (NcException &e) {
            cout << e.what();
            return PLSM_NC_ERROR;
        }
	*/

	int ncid;
        CHK_NCERR(nc_open(fileName,NC_WRITE,&ncid));
	
	int varid;
	
	CHK_NCERR(nc_def_var(ncid,label,NC_INT,0,NULL,&varid));
	
	CHK_NCERR(nc_put_var_int(ncid,varid,pTab));

	CHK_NCERR(nc_put_att_text(ncid,varid,DESCRIPTION.c_str(),1,desc));
	CHK_NCERR(nc_put_att_text(ncid,varid,UNITS.c_str(),1,unit));
		
	CHK_NCERR(nc_close(ncid));
	
	return 0;


	
    }

    Error_t NetCDFManipulator::plsm_add_dimension(const char *fileName, const char *dim_name, int dim_size) {
        /*NcFile dataFile(fname, NcFile::write);

        dataFile.addDim(dim_name, dim_size);

        return 0;
	*/

	int ncid;
        CHK_NCERR(nc_open(fileName,NC_WRITE,&ncid));

	int dimid;

	CHK_NCERR(nc_def_dim(ncid, dim_name, dim_size, &dimid));
	
	CHK_NCERR(nc_close(ncid));

	return 0;

	
    }


    Error_t NetCDFManipulator::plsm_save_attribute(const char *fileName, const char *variable, const char *name, const char *desc) {
        
	/*NcFile dataFile(fileName, NcFile::write);

        try {
            NcVar var;
            var = dataFile.getVar(variable);
            if (var.isNull()) return PLSM_ERROR_FILE_NOT_FOUND;

            var.putAtt(name, desc);

            return 0;
        }
        catch (NcException &e) {
            cout << e.what();
            return PLSM_NC_ERROR;
        }
	*/

	int ncid;
        CHK_NCERR(nc_open(fileName,NC_WRITE,&ncid));

	int varid;
        CHK_NCERR(nc_inq_varid(ncid,variable,&varid));

        char var;
        CHK_NCERR(nc_get_var_text(ncid,varid,&var));
        if (var==0) return PLSM_ERROR_FILE_NOT_FOUND;

	CHK_NCERR(nc_put_att_text(ncid,varid,name,1,desc));
	
	return 0;

    }

}


