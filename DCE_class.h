#ifndef DCE_CLASS_H
#define DCE_CLASS_H
#include "jax_types.h"


class DCE_class {
public:
    explicit DCE_class(jax::expression& expr);
    void apply_dead_code_elimination();

private:
    jax::expression& input_expr;
};



#endif //DCE_CLASS_H
