#ifndef DCE_CLASS_H
#define DCE_CLASS_H
#include "jax_types.h"


class DCE_class {
public:
    explicit DCE_class(jax::expression& expr);
    bool apply_dead_code_elimination();

private:
    jax::expression& input_expr;
};

class CSE_class {
public:
    explicit CSE_class(jax::expression& expr);
    bool apply_common_subexpression_elimination();

private:
    jax::expression& input_expr;
};

class TRS_class {
public:
    explicit TRS_class(jax::expression& expr);
    bool apply_term_rewrite(bool fast_math=true);

private:
    jax::expression& input_expr;
};

class VDN_class {
public:
    explicit VDN_class(jax::expression& expr);
    bool apply_variable_domain_normalisation();

private:
    jax::expression& input_expr;
};





#endif //DCE_CLASS_H
