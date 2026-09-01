#include <ninfer/targets/qwen3_6/decoder_state.h>

#include "ops/kv_cache/d256_profile.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_6 {
namespace {

std::uint32_t page_count(std::uint32_t capacity) {
    if (capacity == 0) { throw std::invalid_argument("Paged KV capacity must be positive"); }
    return 1U + (capacity - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

PagedKVCacheLayout plan_cache(LayoutBuilder& builder, std::uint32_t layers, std::uint32_t capacity,
                              std::int32_t kv_heads, std::int32_t head_dim, KvCacheStorage storage,
                              std::int32_t table_rows, std::uint32_t physical_page_groups) {
    if (layers == 0 ||
        layers > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        kv_heads <= 0 || head_dim <= 0 || table_rows <= 0) {
        throw std::invalid_argument("Paged KV cache geometry is invalid");
    }
    if (head_dim != ops::kD256KVCacheHeadDim) {
        throw std::invalid_argument("Paged KV cache head dimension is not registered");
    }
    const ops::D256KVCacheProfile profile = ops::d256_kv_cache_profile(storage);

    const std::uint32_t logical_pages = page_count(capacity);
    if (physical_page_groups < logical_pages) {
        throw std::invalid_argument("Paged KV physical pages are below logical capacity");
    }

    KVPageGeometry geometry;
    const std::size_t planes_per_layer = 2ULL + static_cast<std::size_t>(profile.key.scaled()) +
                                         static_cast<std::size_t>(profile.value.scaled());
    geometry.planes.reserve(static_cast<std::size_t>(layers) * planes_per_layer);
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        geometry.planes.push_back(
            {profile.key.code_dtype, profile.key.code_leading_extent, kv_heads, 256});
        geometry.planes.push_back(
            {profile.value.code_dtype, profile.value.code_leading_extent, kv_heads, 256});
        if (profile.key.scaled()) {
            geometry.planes.push_back(
                {profile.key.scale_dtype, profile.key.scale_leading_extent, kv_heads, 256});
        }
        if (profile.value.scaled()) {
            geometry.planes.push_back(
                {profile.value.scale_dtype, profile.value.scale_leading_extent, kv_heads, 256});
        }
    }
    return PagedKVCacheLayout{
        .pages = plan_device_kv_page_pool(
            builder, DeviceKVPagePoolSpec{.page_group_count = physical_page_groups,
                                          .geometry         = std::move(geometry)}),
        .execution_tables = plan_kv_execution_tables(
            builder,
            KVExecutionTableSpec{.logical_page_capacity = logical_pages, .table_rows = table_rows}),
        .layers      = layers,
        .max_context = capacity,
        .kv_heads    = kv_heads,
        .head_dim    = head_dim,
        .storage     = storage,
    };
}

} // namespace

DecoderStateLayout plan_decoder_state(LayoutBuilder& builder, const DecoderStateSpec& spec) {
    DecoderStateLayout layout;
    layout.text_kv = plan_cache(builder, spec.full_attention_layers, spec.capacity, spec.kv_heads,
                                spec.attention_head_dim, spec.kv_storage, spec.kv_table_rows,
                                spec.text_physical_page_groups);
    if (spec.enable_mtp) {
        layout.mtp_kv = plan_cache(builder, spec.mtp_layers, spec.capacity, spec.kv_heads,
                                   spec.attention_head_dim, spec.kv_storage, spec.kv_table_rows,
                                   spec.mtp_physical_page_groups);
    }
    return layout;
}

PagedKVCache::PagedKVCache(DeviceSpan backing, const PagedKVCacheLayout& layout)
    : pages_(backing, layout.pages), execution_tables_(backing, layout.execution_tables, pages_),
      layers_(layout.layers), max_context_(layout.max_context), kv_heads_(layout.kv_heads),
      head_dim_(layout.head_dim), storage_(layout.storage) {}

PagedKVCacheView::PagedKVCacheView(const PagedKVCache& cache, Tensor block_table) noexcept
    : cache_(&cache), block_table_(block_table) {}

std::uint32_t PagedKVCacheView::max_context() const noexcept {
    return cache_ == nullptr ? 0 : cache_->max_context();
}

PagedKVLayerView PagedKVCacheView::layer_view(std::uint32_t layer) const {
    if (cache_ == nullptr) { throw std::logic_error("Paged KV execution view is empty"); }
    return cache_->layer_view(layer, block_table_);
}

PagedKVCacheView PagedKVCache::execution_view(const KVExecutionRowLease& row) const {
    if (!row.belongs_to(execution_tables_)) {
        throw std::invalid_argument("Paged KV execution row belongs to another cache");
    }
    return PagedKVCacheView(*this, execution_tables_.row(row.handle()));
}

PagedKVLayerView PagedKVCache::layer_view(std::uint32_t layer, Tensor block_table) const {
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    if (head_dim_ != ops::kD256KVCacheHeadDim && storage_ != KvCacheStorage::BFloat16) {
        throw std::logic_error("Only DFlash BF16 may use a non-D256 paged KV layout");
    }
    const ops::D256KVCacheProfile profile =
        head_dim_ == ops::kD256KVCacheHeadDim
            ? ops::d256_kv_cache_profile(storage_)
            : ops::D256KVCacheProfile{{DType::BF16, head_dim_, DType::U8, 0, 0},
                                      {DType::BF16, head_dim_, DType::U8, 0, 0}};
    const std::size_t stride = 2ULL + static_cast<std::size_t>(profile.key.scaled()) +
                               static_cast<std::size_t>(profile.value.scaled());
    const std::size_t base          = static_cast<std::size_t>(layer) * stride;
    const std::size_t k_scale_index = base + 2;
    const std::size_t v_scale_index =
        k_scale_index + static_cast<std::size_t>(profile.key.scaled());
    return PagedKVLayerView{
        .k_pages       = pages_.plane(base),
        .v_pages       = pages_.plane(base + 1),
        .k_scale_pages = profile.key.scaled() ? pages_.plane(k_scale_index) : Tensor(),
        .v_scale_pages = profile.value.scaled() ? pages_.plane(v_scale_index) : Tensor(),
        .block_table   = block_table,
        .head_dim      = head_dim_,
        .num_kv_heads  = kv_heads_,
        .storage       = storage_,
    };
}

PagedKVBatchLayerView PagedKVCache::batch_layer_view(std::uint32_t layer) const {
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    if (head_dim_ != ops::kD256KVCacheHeadDim && storage_ != KvCacheStorage::BFloat16) {
        throw std::logic_error("Only DFlash BF16 may use a non-D256 paged KV layout");
    }
    const ops::D256KVCacheProfile profile =
        head_dim_ == ops::kD256KVCacheHeadDim
            ? ops::d256_kv_cache_profile(storage_)
            : ops::D256KVCacheProfile{{DType::BF16, head_dim_, DType::U8, 0, 0},
                                      {DType::BF16, head_dim_, DType::U8, 0, 0}};
    const std::size_t stride = 2ULL + static_cast<std::size_t>(profile.key.scaled()) +
                               static_cast<std::size_t>(profile.value.scaled());
    const std::size_t base          = static_cast<std::size_t>(layer) * stride;
    const std::size_t k_scale_index = base + 2;
    const std::size_t v_scale_index =
        k_scale_index + static_cast<std::size_t>(profile.key.scaled());
    return PagedKVBatchLayerView{
        .k_pages       = pages_.plane(base),
        .v_pages       = pages_.plane(base + 1),
        .k_scale_pages = profile.key.scaled() ? pages_.plane(k_scale_index) : Tensor(),
        .v_scale_pages = profile.value.scaled() ? pages_.plane(v_scale_index) : Tensor(),
        .block_tables  = execution_tables_.matrix(),
        .head_dim      = head_dim_,
        .num_kv_heads  = kv_heads_,
        .storage       = storage_,
    };
}

std::size_t DecoderStateLayout::kv_payload_bytes() const noexcept {
    return text_kv.payload_bytes() + (mtp_kv ? mtp_kv->payload_bytes() : 0);
}

DecoderState::DecoderState(DeviceSpan backing, const DecoderStateLayout& layout)
    : text_kv(backing, layout.text_kv) {
    if (layout.mtp_kv) { mtp_kv.emplace(backing, *layout.mtp_kv); }
}

PagedKVCache* DecoderState::mtp_cache() noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

const PagedKVCache* DecoderState::mtp_cache() const noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

} // namespace ninfer::targets::qwen3_6
