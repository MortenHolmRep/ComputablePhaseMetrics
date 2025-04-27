#include <string.h>
#include <Python.h>

unsigned int lz77(const char *S) {
    unsigned int len = strlen(S);
    unsigned int complexity = 1;
    unsigned int ind = 1;
    unsigned int inc = 0;
    unsigned int max = 0;
    unsigned int i = 0;
    while (ind + inc < len) {
        if (S[i + inc] == S[ind + inc]) {
            inc++;
        }
        else{
            if (inc > max) {
                max = inc;
            }
            i++;
            inc = 0;
            if (i == ind) {
                complexity++;
                ind += max + 1;
                max = 0;
                i = 0;
            }
        }
    }
    return (inc != 0) ? complexity + 1 : complexity;
}

// Python wrapper for the lz77 method:
static PyObject *lz77_py(PyObject *self, PyObject *args) {
    const char *binaryStr;

    if (!PyArg_ParseTuple(args, "s", &binaryStr)) {
        return NULL;
    }

    return Py_BuildValue("i", lz77(binaryStr));
}


static PyObject *version(PyObject *self) {
    return Py_BuildValue("s", "Version 1.0");
}


static PyMethodDef LempelZivMethods[] = {
    {"lz77", lz77_py, METH_VARARGS, "Computes the Lempel-Ziv 77 complexity."},
    {"version", (PyCFunction)version, METH_NOARGS, "Returns the version."},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef LempelZivModule = {
    PyModuleDef_HEAD_INIT,
    "LempelZivModule",
    "Python interface for the LZ77 C library function",
    -1,
    LempelZivMethods
};


PyMODINIT_FUNC PyInit_LempelZivModule(void) {
    return PyModule_Create(&LempelZivModule);
}

