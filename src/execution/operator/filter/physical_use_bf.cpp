#include "duckdb/execution/operator/filter/physical_use_bf.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "duckdb/optimizer/predicate_transfer/hash_filter/hash_filter_use_kernel.hpp"
#include "duckdb/optimizer/predicate_transfer/bloom_filter/bloom_filter_use_kernel.hpp"
#include "duckdb/parallel/meta_pipeline.hpp"
#include "duckdb/parallel/pipeline.hpp"
#include "duckdb/common/to_string.hpp"
#include <string>

namespace duckdb {
#ifdef UseHashFilter
PhysicalUseBF::PhysicalUseBF(vector<LogicalType> types, vector<shared_ptr<HashFilter>> bf, idx_t estimated_cardinality)
#else
PhysicalUseBF::PhysicalUseBF(vector<LogicalType> types, vector<shared_ptr<BlockedBloomFilter>> bf, idx_t estimated_cardinality)
#endif
    : CachingPhysicalOperator(PhysicalOperatorType::USE_BF, std::move(types), estimated_cardinality), bf_to_use(bf) {}
    
unique_ptr<OperatorState> PhysicalUseBF::GetOperatorState(ExecutionContext &context) const {
	return make_uniq<CachingOperatorState>();
}

OperatorResultType PhysicalUseBF::ExecuteInternal(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                  GlobalOperatorState &gstate, OperatorState &state_p) const {
	auto &state = state_p.Cast<CachingOperatorState>();
	idx_t row_num = input.size();
    idx_t result_count = input.size();
	SelectionVector sel(STANDARD_VECTOR_SIZE);
	auto bf = bf_to_use[0];
#ifdef UseHashFilter
	HashFilterUseKernel::filter(input.data, bf, sel, result_count, row_num);
#else
	BloomFilterUseKernel::filter(input.data, bf, sel, result_count, row_num);
#endif
	if (result_count == row_num) {
		// nothing was filtered: skip adding any selection vectors
		chunk.Reference(input);
	} else {
		chunk.Slice(input, sel, result_count);
	}
	return OperatorResultType::NEED_MORE_INPUT;
}

static std::string toHexString(long int number) {
    std::stringstream ss;
    ss << std::hex << number;
    return ss.str();
}

string PhysicalUseBF::ParamsToString() const {
	string result;
	auto bf = bf_to_use[0];
	result += toHexString((long)bf.get()) + " ";
	// result += to_string(bf->BoundColsApplied.size()) + " " + to_string(bf->column_bindings_applied_.size()) + " ";
	// for (auto binding : bf->BoundColsApplied) {
	// 	result += to_string(binding) + " ";
	// }
	for (auto binding : bf->column_bindings_applied_) {
		result += binding.ToString() + " ";
	}
	// TODO: Also print column bindings of the child, so that we know which columns are used to probe the bloom filter.
	// result += "from column bindings: ";
	// for (auto binding: children[0]->logical_op->GetColumnBindings()) {
	// 	result += binding.ToString() + " ";
	// }
	return result;
}

void PhysicalUseBF::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	op_state.reset();

	auto &state = meta_pipeline.GetState();
	state.AddPipelineOperator(current, *this);
	children[0]->BuildPipelines(current, meta_pipeline);
	for(auto cell : related_create_bf) {
		cell->BuildPipelinesFromRelated(current, meta_pipeline);
	}
}
}