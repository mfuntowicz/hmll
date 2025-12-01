//
// Created by mfuntowicz on 12/1/25.
//

#ifndef HMLL_HMLL_TYPES_H
#define HMLL_HMLL_TYPES_H
/**
 *
 */
enum hmll_status_code
{
    HMLL_SUCCESS = 0,
    HMLL_ALLOCATION_FAILED = 1,
    HMLL_FILE_NOT_FOUND = 10,
    HMLL_FILE_EMPTY = 11,
    HMLL_FILE_MMAP_FAILED = 12,
    HMLL_SAFETENSORS_HEADER_INVALID = 20,
    HMLL_SAFETENSORS_HEADER_TOO_LARGE = 21,
    HMLL_SAFETENSORS_HEADER_INVALID_START_CHAR = 22,
    HMLL_SAFETENSORS_HEADER_JSON_ERROR = 23,
    HMLL_TENSOR_NOT_FOUND = 30,
    HMLL_UNKNOWN_DTYPE = 31,
};

#define HMLL_STATUS_OK HMLL_SUCCESS;

/**
 * @struct hmll_status
 *
 * Represents an error structure containing information about
 * errors encountered during hmll operations.
 *
 * This structure holds the kind of error and an accompanying
 * error message describing the issue.
 *
 * @var kind
 * The type of error that occurred, represented by
 * hmll_error_kind_t enumeration.
 *
 * @var message
 * A pointer to a character array containing a descriptive
 * error message. Called must manage this memory appropriately.
 *
 * @var length
 * Amount of character stored in the `message` one can read safely.
 */
struct hmll_status
{
    enum hmll_status_code what;
    const char* message;
};
typedef struct hmll_status hmll_status_t;

static struct hmll_status HMLL_SUCCEEDED = {HMLL_SUCCESS, nullptr};

/// Helper methods indicating the result of a status
bool hmll_status_succeeded(struct hmll_status status);
bool hmll_status_has_error(struct hmll_status status);

#endif //HMLL_HMLL_TYPES_H
