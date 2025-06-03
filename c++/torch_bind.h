#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct               Model;
struct               Tensor;

//int                  GetTypeId(char const* type);
struct Model*        NewModel(char const* model, char const* options);
void                 FreeModel(struct Model* model);
struct Tensor*       NewTensorInt8(void* data, size_t size, size_t* shape, size_t dims);
struct Tensor*       NewTensorInt16(void* data, size_t size, size_t* shape, size_t dims);
struct Tensor*       NewTensorInt32(void* data, size_t size, size_t* shape, size_t dims);
struct Tensor*       NewTensorInt64(void* data, size_t size, size_t* shape, size_t dims);
struct Tensor*       NewTensorUInt8(void* data, size_t size, size_t* shape, size_t dims);
struct Tensor*       NewTensorUInt16(void* data, size_t size, size_t* shape, size_t dims);
struct Tensor*       NewTensorUInt32(void* data, size_t size, size_t* shape, size_t dims);
struct Tensor*       NewTensorUInt64(void* data, size_t size, size_t* shape, size_t dims);
struct Tensor*       NewTensorFloat32(void* data, size_t size, size_t* shape, size_t dims);
struct Tensor*       NewTensorFloat64(void* data, size_t size, size_t* shape, size_t dims);
void                 FreeTensor(struct Tensor* tensor);
void                 Train(struct Model* model, struct Tensor* data, struct Tensor const** target, int num_targets, int epochs);
struct Tensor*       Infer(struct Model* model, struct Tensor* data);

#ifdef __cplusplus
}
#endif
