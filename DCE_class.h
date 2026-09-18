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

class CSE_class {
public:
    explicit CSE_class(jax::expression& expr);
    void apply_common_subexpression_elimination();

private:
    jax::expression& input_expr;
};

class TRS_class {
public:
    explicit TRS_class(jax::expression& expr);
    void apply_term_rewrite();

private:
    jax::expression& input_expr;
};

class VDR_class {
public:
    explicit VDR_class(jax::expression& expr);
    void apply_varaible_domain_reduction();
};





#endif //DCE_CLASS_H
