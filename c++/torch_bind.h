#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct               basic_model;
struct               tensor;

int                  GetTypeId(char const* type);
struct basic_model*  NewModel(char const* model, char const* options);
void                 FreeModel(struct basic_model* model);
struct tensor*       NewTensorFromBlob(uint8_t* data, int64_t size, int64_t* shape, int64_t dims, int dtype);
void                 FreeTensor(struct tensor* tensor);
void                 Train(struct basic_model* model, struct tensor const* data, struct tensor const* target, int epochs);
size_t               Infer(struct basic_model* model, struct tensor const* data, void* result_buffer, size_t buffer_size);

#ifdef __cplusplus
}
#endif
