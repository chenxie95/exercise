#include <iostream>
#include <string>
#include <random>
#include <math.h>

using namespace std;

float logadd(float loga, float logb)
{
    float max = (loga > logb) ? loga : logb;
    loga = loga - max;
    logb = logb - max;
    float c = exp(loga) + exp(logb);
    return max + log(c);
}

void init_logits(vector<vector<vector<float>>> &logits, int N, int M, int V)
{
    logits.resize(N);
    for (int i = 0; i < N; i++)
    {
        logits[i].resize(M+1);
        for (int j = 0; j < M+1; j++)
        {
            logits[i][j].resize(V);
            float sum = 0;
            for (int k = 0; k < V; k++)
            {
                logits[i][j][k] = rand() *1.0 / (RAND_MAX);
                sum += logits[i][j][k];
            }
            // softmax over all logits
            for (int k = 0; k < V; k++)
            {
                logits[i][j][k] = log(logits[i][j][k] / sum);
            }
        }
    }
}

void init_labels(vector<int> &label, int M, int V)
{
    label.resize(M + 1);
    for (int i = 0; i < M + 1; i++)
    {
        // label will never be 0 (blank)
        label[i] = rand() % (V - 1) + 1;
    }
}

int main(int argc, const char *argv[])
{
    if (argc != 4)
    {
        std::cerr << "usage: ./rnnt N M V\n e.g. ./rnnt 10 154 2000\n";
        return -1;
    }
    int N = stoi(argv[1]);
    int M = stoi(argv[2]);
    int V = stoi(argv[3]);
    // init the logits
    vector<vector<vector<float>>> logits;
    init_logits(logits, N, M, V);

    // init the label
    vector<int> label;
    init_labels(label, M, V);

    // init the forward and backward accumulated likelihood matrices (alpha, beta)
    vector<vector<float>> alpha;
    alpha.resize(N);
    for (int i = 0; i < N; i++)
    {
        alpha[i].resize(M+1);
    }

    for (int i = 0; i < N; i++)
    {
        if (i == 0)
        {
            alpha[i][0] = logits[i][0][0];
        }
        else
        {
            alpha[i][0] = alpha[i - 1][0] + logits[i][0][0];
        }
    }
    for (int i = 0; i < N; i++)
    {
        for (int j = 1; j < M + 1; j++)
        {
            if (i == 0)
            {
                alpha[i][j] = alpha[i][j - 1] + logits[i][j-1][label[j]];
            }
            else
            {
                alpha[i][j] = logadd(alpha[i - 1][j] + logits[i-1][j][0], alpha[i][j - 1] + logits[i][j-1][label[j]]);
            }
        }
    }
    float loss = alpha[N - 1][M] + logits[N-1][M][0];
    printf("Total cost for RNN-T is: %f\n", loss);

#if 1
    // beta: store the errors for the matrix: N x (M+1)
    // beta_rnn: store the gradient for label
    // beta_bland: store the gradient for the blank symbol
    vector<vector<float>> beta, beta_rnn, beta_blank;
    beta.resize(N);
    beta_rnn.resize(N);
    beta_blank.resize(N);
    for (int i = 0; i < N; i++)
    {
        beta[i].resize(M + 1);
        beta_rnn[i].resize(M + 1);
        beta_blank[i].resize(M + 1);
    }
    // back-prop the error in the alpha matrix,
    // which should be eaiser and more suitable for RNN-T presented in the paper.
    for (int i = N - 1; i >= 0; i--)
    {
        for (int j = M; j >= 0; j--)
        {
            if (i == N - 1)
            {
                if (j == M)
                {
                    beta[i][j] = loss + logits[N-1][M][0];
                    beta_rnn[i][j] = loss + alpha[N - 1][M];
                }
                else
                {
                    beta[i][j] = beta[i][j + 1] + logits[i][j][label[j+1]];
                    beta_rnn[i][j] = beta[i][j + 1] + alpha[i][j];
                }

            }
            else if (j == M)
            {
                beta[i][j] = beta[i + 1][j] + logits[i][j][0];
                beta_blank[i][j] = beta[i + 1][j] + alpha[i][j];
            }
            else
            {
                beta[i][j] = logadd(beta[i + 1][j] + logits[i][j][0], beta[i][j + 1] + logits[i][j][label[j + 1]]);
                beta_blank[i][j] = beta[i + 1][j] + alpha[i][j];
                beta_rnn[i][j] = beta[i][j + 1] + alpha[i][j];
            }
        }
    }
#endif
    printf ("Finished the back-prop in the RNN-T matrix for error and gradient computation!\n");
    return 0;
}
