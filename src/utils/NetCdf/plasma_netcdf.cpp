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

	int ncid;
        CHK_NCERR(nc_open(fileName,NC_NOWRITE,&ncid));

	int varid;
        CHK_NCERR(nc_inq_varid(ncid,name,&varid));

        CHK_NCERR(nc_get_var(ncid,varid,array));

	CHK_NCERR(nc_close(ncid))

        return 0;

    }



    Error_t NetCDFManipulator::plsm_get_dim_var(const char *fileName, const char *name, int *dimVar) {

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

	int ncid;
        CHK_NCERR(nc_open(fileName,NC_WRITE,&ncid));

	int dims[3];
	CHK_NCERR(nc_inq_dimid(ncid,"x",&dims[0]));		
	CHK_NCERR(nc_inq_dimid(ncid,"y",&dims[1]));
	CHK_NCERR(nc_inq_dimid(ncid,"z",&dims[2]));

	int varid;
	
	CHK_NCERR(nc_def_var(ncid,label,NC_DOUBLE,3,dims,&varid));

	CHK_NCERR(nc_put_var_double(ncid,varid,pTab));

	CHK_NCERR(nc_put_att_text(ncid,varid,DESCRIPTION.c_str(),strlen(desc),desc));
	CHK_NCERR(nc_put_att_text(ncid,varid,UNITS.c_str(),strlen(unit),unit));
		
	CHK_NCERR(nc_close(ncid));

	return 0;


    }

    Error_t NetCDFManipulator::plsm_save_1D_double_array(const char *fileName, double *pTab, const char *label, const char *dim_label, const char *unit, const char *desc) {

	int ncid;
        CHK_NCERR(nc_open(fileName,NC_WRITE,&ncid));

	int dimid[1];
	CHK_NCERR(nc_inq_dimid(ncid,dim_label,&dimid[0]));

	int varid;
	
	CHK_NCERR(nc_def_var(ncid,label,NC_DOUBLE,1,dimid,&varid));

	CHK_NCERR(nc_put_var_double(ncid,varid,pTab));

	CHK_NCERR(nc_put_att_text(ncid,varid,DESCRIPTION.c_str(),strlen(desc),desc));
	CHK_NCERR(nc_put_att_text(ncid,varid,UNITS.c_str(),strlen(unit),unit));
		
	CHK_NCERR(nc_close(ncid));
	
	return 0;
	
    }

    Error_t NetCDFManipulator::plsm_save_double(const char *fileName, double *pTab, const char *label, const char *unit, const char *desc) {

	int ncid;
        CHK_NCERR(nc_open(fileName,NC_WRITE,&ncid));

	int varid;
	
	CHK_NCERR(nc_def_var(ncid,label,NC_DOUBLE,0,NULL,&varid));
	
	CHK_NCERR(nc_put_var_double(ncid,varid,pTab));

	CHK_NCERR(nc_put_att_text(ncid,varid,DESCRIPTION.c_str(),strlen(desc),desc));
	CHK_NCERR(nc_put_att_text(ncid,varid,UNITS.c_str(),strlen(unit),unit));
		
	CHK_NCERR(nc_close(ncid));
	
	return 0;



	
    }

    Error_t NetCDFManipulator::plsm_save_int(const char *fileName, int *pTab, const char *label, const char *unit, const char *desc) {

	int ncid;
        CHK_NCERR(nc_open(fileName,NC_WRITE,&ncid));
	
	int varid;
	
	CHK_NCERR(nc_def_var(ncid,label,NC_INT,0,NULL,&varid));
	
	CHK_NCERR(nc_put_var_int(ncid,varid,pTab));

	CHK_NCERR(nc_put_att_text(ncid,varid,DESCRIPTION.c_str(),strlen(desc),desc));
	CHK_NCERR(nc_put_att_text(ncid,varid,UNITS.c_str(),strlen(unit),unit));
		
	CHK_NCERR(nc_close(ncid));
	
	return 0;


	
    }

    Error_t NetCDFManipulator::plsm_add_dimension(const char *fileName, const char *dim_name, int dim_size) {

	int ncid;
        CHK_NCERR(nc_open(fileName,NC_WRITE,&ncid));

	int dimid;

	CHK_NCERR(nc_def_dim(ncid, dim_name, dim_size, &dimid));
	
	CHK_NCERR(nc_close(ncid));

	return 0;

	
    }


    Error_t NetCDFManipulator::plsm_save_attribute(const char *fileName, const char *variable, const char *name, const char *desc) {

	int ncid;
        CHK_NCERR(nc_open(fileName,NC_WRITE,&ncid));

	int varid;
        CHK_NCERR(nc_inq_varid(ncid,variable,&varid));

        char var;
        CHK_NCERR(nc_get_var_text(ncid,varid,&var));
        if (var==0) return PLSM_ERROR_FILE_NOT_FOUND;

	CHK_NCERR(nc_put_att_text(ncid,varid,name,strlen(desc),desc));
	
	return 0;

    }

}


