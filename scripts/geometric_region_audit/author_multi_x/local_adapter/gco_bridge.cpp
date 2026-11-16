// Local C ABI wrapper for the unmodified graph-cut optimizer shipped by Multi-X.
#include <cfloat>
#include <cstring>
#include <exception>
#include <vector>
#include "GCoptimization.h"

extern "C" int multix_label(int n, int k, float* unary, int e, int* edges,
                            float spatial, float* costs, int* labels,
                            double* energy, char* error) {
    try {
        GCoptimizationGeneralGraph gc(n, k);
        gc.setDataCost(unary);
        std::vector<float> smooth(k*k, spatial);
        for(int j=0;j<k;j++) smooth[j*k+j]=0;
        gc.setSmoothCost(smooth.data());
        gc.setLabelCost(costs);
        for(int j=0;j<e;j++) gc.setNeighbors(edges[2*j],edges[2*j+1]);
        for(int j=0;j<n;j++) gc.setLabel(j, labels[j]);
        int iterations=0;
        *energy=gc.expansion(iterations, 1000);
        for(int j=0;j<n;j++) labels[j]=gc.whatLabel(j);
        return 0;
    } catch(GCException &ex) { std::strncpy(error,ex.message,999);return 1; }
      catch(std::exception &ex) { std::strncpy(error,ex.what(),999);return 2; }
      catch(...) { std::strncpy(error,"unknown graph cut error",999);return 3; }
}
