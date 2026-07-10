//
// Created by Yuriy Kulinchenko on 01/07/2026.
//

#include "DCE_class.h"

#include <unordered_set>

using namespace jax;

/*

Architecture of DCE pass:

The purpose of this pass is dead code elimination, which will remove
all unused equations in a jaxpr. This is an in-place transformation,
so it will overwrite the passed jaxpr.

DCE works by traversing backwards from the outputs of a given jaxpr,
and noting which equation results are reachable. Equation results
which are unreachable can be eliminated, which is done in the final pass

As the graph is traversed backwards, a 'keep' mapping can be maintained,
which tracks which equations need to be kept.

*/

DCE_class::DCE_class(expression& expr): input_expr(expr) {}

void DCE_class::apply_dead_code_elimination() {
    auto& equations = input_expr.equations;

    std::vector keep(equations.size(), false);
    std::unordered_set<size_t> used_ids {};

    for (auto& val: input_expr.outvals) {
        if (val.is<literal_t>()) continue;
        used_ids.insert(val.get_var().get_id());
    }


    for (size_t i = equations.size(); i--> 0;) {
        // Step 1: determine whether the equation should be kept,
        // based on whether it contains an output in used_ids:

        auto& equation = equations[i];
        bool keep_equation = false;

        for (auto& output: equation.get_output()) {
            if (used_ids.contains(output.get_id())) {
                keep_equation = true;
                keep[i] = true;
                break;
            }
        }

        // Step 2: if the equation should be kept, place all inputs
        // into used_ids:

        if (!keep_equation) continue;
        for (auto& input: equation.get_input()) {
            if (input.is<literal_t>()) continue;
            used_ids.insert(input.get_var().get_id());
        }

        // Step 3: if the equation has children, recursively apply DCE:

        switch (equation.get_op()) {
            using enum primitive_op;
            case COND: {
                for (auto& expr: std::get<cond_params>(equation.get_params()).branches) {
                    expr.eliminate_dead_code();
                }
                break;
            }
            default:
        }
    }

    // Finally, iterate through keep vector, and remove dead equations:

    size_t place_index = 0;
    for (size_t i = 0; i < equations.size(); i++) {
        if (!keep[i]) continue;
        if (place_index != i) equations[place_index] = std::move(equations[i]);
        place_index++;
    }

    // Shrink the vector appropriately:
    equations.erase(equations.begin() + place_index, equations.end());
}

