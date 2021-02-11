#include <errno.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define MLP_ENOMEM 1
#define MLP_EFILE 2
#define MLP_EFILEFORMAT 3

struct mlp {
    float *input_neurons;
    float *input_hidden_weights;
    float *hidden_neurons;
    float *hidden_output_weights;
    float *output_neurons;
    float *hidden_changes;
    float *output_changes;
    int n_input;
    int n_hidden;
    int n_output;
};

struct mlp_parameters {
    double min_weight;
    double max_weight;
};

struct mlp_parameters mlp_defaults = {
    .min_weight = -2,
    .max_weight = 2
};

void shuffle(int *arr, int n)
{
	for (int i = n; i-- > 1; ) {
		int to = i + 1;
		int j =  rand() % to;
                int t = arr[i];
                arr[i] = arr[j];
                arr[j] = t;
	}
}


void
print_matrix(const float matrix[], int n, int m)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++)
            printf("%f\t", matrix[i * m + j]);
        puts("");
    }
}

void
print_vector(const float vec[], int n)
{
    for (int i = 0; i < n; i++)
            printf("%f\n", vec[i]);
}

float rand_range(float from, float to)
{
    return ((float) rand() / (float) RAND_MAX) * (to - from) + from;
}

float sigmoid(float f)
{
    return 1 / (1 + exp(-f));
}

float mse(const float* restrict v1, const float* restrict v2, int n)
{
    int i;
    float e = 0.0, r, d;

#pragma omp parallel for private(d) reduction(+:e)
    for (i = 0; i < n; i++) {
        d = v1[i] - v2[i];
        e += d * d;
    }

    r = e * (1.0 /  (float) (2 * n));

    return r;
}

void
sigmoid_vector(float* restrict vec, int n)
{
#pragma omp parallel for
    for (int i = 0; i < n; i++)
        vec[i] = sigmoid(vec[i]);
}

void
mlp_neurons_input(float* restrict input, int n,
                 const float* restrict weights, int m,
                 float * result)
{
    int i, j, k;
    float total;
#pragma omp parallel for private(total, i)
    for (i = 0; i < m; i++) {
        total = 0.0;
#pragma omp parallel for private(j, k) reduction(+:total)
        for (j = 0; j < n; j++) {
            k = j * m + i;
            total += input[j] * weights[k];
        }
        result[i] = total;
    }
}

int
mlp_create(struct mlp *m, const struct mlp_parameters *parms,
           int input, int hidden, int output)
{
    m->n_input = input;
    m->n_hidden = hidden;
    m->n_output = output;

    // Allocate neurons
    m->input_neurons = malloc(sizeof(float) * (input + 1));
    if (m->input_neurons == NULL) {
        fprintf(stderr, "Cannot allocate mlp input neurons %s %d\n",
                __FILE__, __LINE__);
        return MLP_ENOMEM;
    }
    m->input_neurons[m->n_input] = -1.0; // bias

    m->hidden_neurons = malloc(sizeof(float) * (hidden + 1));
    if (m->hidden_neurons == NULL) {
        fprintf(stderr, "Cannot allocate mlp hidden neurons %s %d\n",
                __FILE__, __LINE__);
        return MLP_ENOMEM;
    }
    m->hidden_neurons[m->n_hidden] = -1.0; // bias

    m->output_neurons = malloc(sizeof(float) * (output));
    if (m->hidden_neurons == NULL) {
        fprintf(stderr, "Cannot allocate mlp output neurons %s %d\n",
                __FILE__, __LINE__);
        return MLP_ENOMEM;
    }

    // Allocate weights
    m->input_hidden_weights = malloc(sizeof(float) * (input + 1) * hidden );
    if (m->input_hidden_weights == NULL) {
        fprintf(stderr, "Cannot allocate mlp input-hidden weights %s %d\n",
                __FILE__, __LINE__);
        return MLP_ENOMEM;
    }
    for (int i = 0; i < (input + 1) * hidden; i++)
        m->input_hidden_weights[i] = rand_range(parms->min_weight, parms->max_weight);
    m->hidden_changes = calloc((m->n_input + 1) * m->n_hidden, sizeof(float));
    if (m->hidden_changes == NULL) {
        fprintf(stderr, "Cannot allocate hidden deltas %s %d\n",
                __FILE__, __LINE__);
        return MLP_ENOMEM;
    }


    m->hidden_output_weights = malloc(sizeof(float) * (hidden + 1) * output);
    if (m->hidden_output_weights == NULL) {
        fprintf(stderr, "Cannot allocate mlp hidden-output layer %s %d\n",
                __FILE__, __LINE__);
        return MLP_ENOMEM;
    }
    for (int i = 0; i < (hidden + 1) * output; i++)
        m->hidden_output_weights[i] = rand_range(parms->min_weight, parms->max_weight);
    // Storage for deltas for back propagation momentum term
    m->output_changes = calloc((m->n_hidden + 1) * m->n_output, sizeof(float));
    if (m->output_changes == NULL) {
        fprintf(stderr, "Cannot allocate output deltas %s %d\n",
                __FILE__, __LINE__);
        return MLP_ENOMEM;
    }

    return 0;
}

void
mlp_classify(const struct mlp *m, const float *pattern)
{
    memcpy(m->input_neurons, pattern, m->n_input * sizeof(float));
    mlp_neurons_input(m->input_neurons, m->n_input + 1,
                      m->input_hidden_weights, m->n_hidden,
                      m->hidden_neurons);
    sigmoid_vector(m->hidden_neurons, m->n_hidden);

    mlp_neurons_input(m->hidden_neurons, m->n_hidden + 1,
                      m->hidden_output_weights, m->n_output,
                      m->output_neurons);
    sigmoid_vector(m->output_neurons, m->n_output);
}

int
mlp_train_single(struct mlp *m, const float *pattern, const float *target,
                 float eta, float momentum)
{
    float out, deriv, err, delta, change, *errors;
    int i, j, k;

    mlp_classify(m, pattern);

    // Update hidden to output

    errors = calloc(m->n_hidden + 1, sizeof(float));
    if (errors == NULL) {
        fprintf(stderr, "Cannot allocate error deltas %s %d\n",
                __FILE__, __LINE__);
        return MLP_ENOMEM;
    }

    for (i = 0; i < (m->n_output); i++) {
        out = m->output_neurons[i];
        err = target[i] - out;
        deriv = out * (1 - out);
        delta = deriv * err;
        for (j = 0; j < m->n_hidden + 1; j++) {
            k = j * m->n_output + i;
            errors[j] += m->hidden_output_weights[k] * delta;
            change = delta * eta * m->hidden_neurons[j];
            m->hidden_output_weights[k] +=
                change + momentum * m->output_changes[k];
            m->output_changes[k] = change;
        }
    }

    // Update input to hidden
    for (i = 0; i < m->n_hidden; i++) {
        out = m->hidden_neurons[i];
        deriv = out * (1 - out);
        delta = deriv * errors[i];
        for (j = 0; j < m->n_input + 1; j++) {
            k = j * m->n_hidden + i;
            change = delta * eta * m->input_neurons[j];
            m->input_hidden_weights[k] +=
                change + momentum * m->hidden_changes[k];
            m->hidden_changes[k] = change;
        }
    }

    free(errors);
    return 0;
}

int
mlp_train(struct mlp *m, const float *patterns[],
          const float *targets[], int n, int iterations,
          float eta, float momentum)
{
    int i, j, k_p, k_t, indices[n], ret = 0;

    for (i = 0; i < n; i++)
        indices[i] = i;

    printf("%d\n", iterations);
    for (i = 0; i < iterations; i++) {
        // shuffle(indices, n);
        for (j = 0; j < n; j++) {
            k_p = m->n_input * indices[j] * sizeof(float);
            k_t = m->n_output * indices[j] * sizeof(float);
            ret += mlp_train_single(m, (void *) patterns + k_p,
                                    (void *) targets + k_t, eta, momentum);
        }
        if (ret) return ret;
    }
    puts("DONE");
    return 0;
}

void
mlp_train_batch(struct mlp *m, const float *pattern[],
                const float *target[], int n,
                float eta, float momentum)
{

}

int
mlp_save(const struct mlp *m, const char *filename)
{
    FILE *f = fopen(filename, "w");

    if (f == NULL) {
        fprintf(stderr, "Can't open weight file for writing.\n");
        return MLP_EFILE;
    }
    fprintf(f, "%d %d %d\n", m->n_input, m->n_hidden, m->n_output);

    for (int i = 0; i < (m->n_input+1); i++) {
        for (int j = 0; j < m->n_hidden; j++) {
            int k = i * m->n_hidden + j;
            fprintf(f, "%f\n", m->input_hidden_weights[k]);
        }
    }

    for (int i = 0; i < (m->n_hidden+1); i++) {
        for (int j = 0; j < m->n_output; j++) {
            int k = i * m->n_output + j;
            fprintf(f, "%f\n", m->hidden_output_weights[k]);
        }
    }
    fclose(f);
    return 0;
}

int
mlp_load(struct mlp *m, const char *filename)
{
    FILE *f = fopen(filename, "r");
    int ret;

    if (f == NULL) {
        fprintf(stderr, "Can't open weight file for reading.\n");
        return MLP_EFILE;
    }
    fscanf(f, "%d %d %d", &m->n_input, &m->n_hidden, &m->n_output);

    ret = mlp_create(m, &mlp_defaults, m->n_input, m->n_hidden, m->n_output);
    if (ret) {
        fprintf(stderr, "Error creating mlp in load.\n");
        return ret;
    }

    for (int i = 0; i < (m->n_input+1); i++) {
        for (int j = 0; j < m->n_hidden; j++) {
            int k = i * m->n_hidden + j;
            float r;
            ret = fscanf(f, "%f", &r);
            if (ret != 1) {
                fprintf(stderr, "Not enough weights in file\n");
                fclose(f);
                return MLP_EFILEFORMAT;
            }
            m->input_hidden_weights[k] = r;
        }
    }

    for (int i = 0; i < (m->n_hidden+1); i++) {
        for (int j = 0; j < m->n_output; j++) {
            float r;
            int k = i * m->n_output + j;
            ret = fscanf(f, "%f", &r);
            if (ret != 1) {
                fprintf(stderr, "Not enough weights in file\n");
                fclose(f);
                return MLP_EFILEFORMAT;
            }
            m->hidden_output_weights[k] = r;
        }
    }
    fclose(f);
    return 0;
}


void
mlp_print(const struct mlp *m)
{
    puts("Input-Hidden");
    for (int i = 0; i < (m->n_input+1); i++) {
        for (int j = 0; j < m->n_hidden; j++) {
            int k = i * m->n_hidden + j;
            printf("%d-%d (%d): %.2f ", i, j, k, m->input_hidden_weights[k]);
        }
    }
    puts("\nHidden-output");
    for (int i = 0; i < (m->n_hidden+1); i++) {
        for (int j = 0; j < m->n_output; j++) {
            int k = i * m->n_output + j;
            printf("%d-%d (%d): %.2f ", i, j, k, m->hidden_output_weights[k]);
        }
    }
    puts("");
}

void
mlp_destroy(struct mlp *m)
{
    free(m->input_neurons);
    free(m->hidden_neurons);
    free(m->output_neurons);
    free(m->input_hidden_weights);
    free(m->hidden_output_weights);
    free(m->hidden_changes);
    free(m->output_changes);
}

int main(int argc, char *arv[])
{
    struct mlp m;
    int ret;

    float eta = 0.2;
    float momentum = 1.0;
    int iterations = 1000;

    const float p[4][2] = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    const float t[4][1] = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    puts("CREATE");
    ret = mlp_create(&m, &mlp_defaults, 2, 2, 1);
    if (ret) {
        fprintf(stderr, "Error creating mlp.\n");
        exit(EXIT_FAILURE);
    }

    mlp_save(&m, "xor_before.n");
    puts("CLASSIFY BEFORE TRAINING");
    mlp_classify(&m, p[0]);
    printf("p0: %f \n", m.output_neurons[0]);
    mlp_classify(&m, p[1]);
    printf("p1: %f \n", m.output_neurons[0]);
    mlp_classify(&m, p[2]);
    printf("p2: %f \n", m.output_neurons[0]);
    mlp_classify(&m, p[3]);
    printf("p3: %f \n", m.output_neurons[0]);

    puts("PRINT");
    mlp_print(&m);

    puts("TRAINING");
    ret = mlp_train(&m, (const float **) p, (const float **) t,
                    4, iterations, eta, momentum);
    if (ret) return EXIT_FAILURE;
    mlp_classify(&m, p[0]);
    printf("p0: %f \n", m.output_neurons[0]);
    mlp_classify(&m, p[1]);
    printf("p1: %f \n", m.output_neurons[0]);
    mlp_classify(&m, p[2]);
    printf("p2: %f \n", m.output_neurons[0]);
    mlp_classify(&m, p[3]);
    printf("p3: %f \n", m.output_neurons[0]);

    puts("SAVE out.n");
    mlp_save(&m, "out.n");
    mlp_destroy(&m);

    puts("LOAD xor.n");
    mlp_load(&m, "xor.nn");

    puts("XOR");
    mlp_print(&m);
    mlp_classify(&m, p[0]);
    printf("p0: %f \n", m.output_neurons[0]);
    mlp_classify(&m, p[1]);
    printf("p1: %f \n", m.output_neurons[0]);
    mlp_classify(&m, p[2]);
    printf("p2: %f \n", m.output_neurons[0]);
    mlp_classify(&m, p[3]);
    printf("p3: %f \n", m.output_neurons[0]);
    mlp_destroy(&m);

    //srand(time(NULL));

    /*
    puts("STRESS");
    const int num_inputs = 50000;
    const int num_hidden = 10000;
    const int num_outputs = 50;
    iterations = 1;

    ret = mlp_create(&m, &mlp_defaults, num_inputs, num_hidden, num_outputs);
    if (ret) {
        fprintf(stderr, "Error creating mlp.\n");
        exit(EXIT_FAILURE);
    }
    float p5[num_inputs];
    for (int i = 0; i < iterations; i++) {
        for (int j = 0; j < num_inputs; j++)
            p5[j] = (float) rand() / (float) RAND_MAX;
        mlp_classify(&m, p5);
        printf("p5: ");
        for (int i = 0; i < num_outputs; i++)
            printf("%f ", m.output_neurons[i]);
        puts("");
    }
    mlp_destroy(&m);
    */
    return EXIT_SUCCESS;
}
