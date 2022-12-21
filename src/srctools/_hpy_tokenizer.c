#include "hpy.h"
//from cpython.mem cimport PyMem_Free, PyMem_Malloc, PyMem_Realloc
#include <stdint.h>
#include <string.h>

static HPyGlobal g_Token;
static HPyGlobal g_TokenSyntaxError;

typedef struct {
    HPy_ssize_t line_num;
	HPyField error_type;

    HPyField filename;
    HPyField pushback_tok;
    HPyField pushback_val;

    unsigned char string_brackets: 1;
    unsigned char allow_escapes: 1;
    unsigned char colon_operator: 1;
    unsigned char allow_star_comments: 1;
    unsigned char file_input: 1;
    unsigned char last_was_cr: 1;
} BaseTokenizer;

HPyType_HELPERS(BaseTokenizer)

HPyDef_SLOT(BaseTokenizer_new, BaseTokenizer_traverse_impl, HPy_tp_traverse)
static HPy Point_new_impl (void *self_obj, HPyFunc_visitproc visit, void *arg) {
	BaseTokenizer *self = (BaseTokenizer *)self_obj;
	HPy_Visit(&self->error_type);
	HPy_Visit(&self->filename);
	HPy_Visit(&self->pushback_tok);
	HPy_Visit(&self->pushback_val);
	return 0;
}

HPyDef_SLOT(BaseTokenizer_new, BaseTokenizer_init_impl, HPy_tp_init)
int BaseTokenizer_init_impl (HPyContext *ctx, HPy self, HPy *args, HPy_ssize_t nargs, HPy kw)
{
	static const char *kwlist[] = {"filename", "error", NULL};
	BaseTokenizer *tok = BaseTokenizer_AsStruct(ctx, self);
    HPyTracker ht;
    HPy filename = HPy_NULL;
    HPy error = HPy_NULL;
    HPy os_mod = HPy_NULL;
    HPy fspath = HPy_NULL;
    HPy filename_repr = HPy_NULL;
    HPy args = HPy_NULL;
    if (!HPyArg_ParseKeywords(ctx, &ht, args, nargs, kw, "OO", kwlist, &filename, &error))
        return HPy_NULL;

    if (HPy_Is(filename, ctx->h_None)) {
    	HPyField_Store(ctx, self, &tok, ctx->h_None);
    } else {
    	os_mod = HPyImport_ImportModule(ctx, "os");
    	if (HPy_IsNull(ctx, os_mod)) {
    		goto error;
    	}
    	fspath = HPy_GetAttr_s(ctx, os_mod, "fspath");
    	HPy_Close(ctx, os_mod);
    	if (HPy_IsNull(ctx, fspath)) {
    		goto error;
    	}
    	args = HPy_BuildValue(ctx, "(O)", filename);
    	if (HPy_IsNull(ctx, args)) {
    		HPy_Close(ctx, fspath);
    		goto error;
    	}
    	fname = HPy_CallTupleDict(ctx, fspath, args, HPy_NULL);
    	HPy_Close(ctx, args);
    	HPy_Close(ctx, fspath);
    	if (HPy_IsNull(ctx, fname)) {
    		goto error;
    	}
    	if (HPy_TypeCheck(ctx, fname, ctx->h_BytesType)) {
    		HPy filename_repr =
    	}
    }

	HPyTracker_close(ctx, ht);
    return 0;

    error:
	HPyTracker_close(ctx, ht);
    return HPy_NULL;
}

HPyDef_SLOT(Point_repr, Point_repr_impl, HPy_tp_repr)
static HPy Point_repr_impl(HPyContext *ctx, HPy self)
{
    PointObject *point = PointObject_AsStruct(ctx, self);
    char msg[256];
    snprintf(msg, 256, "Point(%g, %g)", point->x, point->y);
    return HPyUnicode_FromString(ctx, msg);
    //return HPyUnicode_FromFormat("Point(%g, %g)", point->x, point->y);
}


static HPyDef *basetokenizer_defines[] = {
    &BaseTokenizer_init,
    &Point_repr,
    NULL,
};
static HPyType_Spec basetokenizer_spec = {
    .name = "srctools.tokenizer.BaseTokenizer",
    .basicsize = sizeof(BaseTokenizer),
    .flags = HPy_TPFLAGS_DEFAULT,
    .defines = basetokenizer_defines,
};

static HPyDef *module_defines[] = {
    NULL
};
static HPyModuleDef moduledef = {
    .name = "_hpy_vtf_readwrite",
    .doc = "Functions for reading/writing VTF data.",
    .size = -1,
    .defines = module_defines
};

HPy_MODINIT(_hpy_vtf_readwrite)
static HPy init__hpy_vtf_readwrite_impl(HPyContext *ctx)
{
    HPy m, h_basetokenizer;
    m = HPyModule_Create(ctx, &moduledef);
    if (HPy_IsNull(m))
        return HPy_NULL;
    h_basetokenizer = HPyType_FromSpec(ctx, &basetokenizer_spec, NULL);
    if (HPy_IsNull(h_basetokenizer))
      return HPy_NULL;
    HPy_SetAttr_s(ctx, m, "BaseTokenizer", h_basetokenizer);
    return m;
}
