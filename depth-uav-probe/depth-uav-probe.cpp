#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <cstdio>
#include <cstring>
#include <cstdint>

using Microsoft::WRL::ComPtr;

static void result(const char* name, HRESULT hr) {
    std::printf("%s: 0x%08lx\n", name, static_cast<unsigned long>(hr));
}

static bool ok(const char* name, HRESULT hr) {
    if (FAILED(hr)) result(name, hr);
    return SUCCEEDED(hr);
}

static int coherence_test(ID3D12Device* dev, const D3D12_RESOURCE_DESC& combined, const D3D12_HEAP_PROPERTIES& heap, bool shared) {
    constexpr UINT slices = 6;
    auto depth_desc = combined;
    auto uav_desc = combined;
    if (!shared) {
        depth_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        uav_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }
    ComPtr<ID3D12Resource> depth, uav, readback;
    if (!ok("Depth creation", dev->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &depth_desc,
        shared ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_DEPTH_WRITE,
        nullptr, IID_PPV_ARGS(&depth)))) return 2;
    if (shared) uav = depth;
    else if (!ok("UAV creation", dev->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &uav_desc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&uav)))) return 2;

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprints[2][slices]{};
    UINT64 forward_bytes{}, bytes{};
    dev->GetCopyableFootprints(&uav_desc, 0, slices, 0, footprints[0], nullptr, nullptr, &forward_bytes);
    dev->GetCopyableFootprints(&depth_desc, 0, slices, (forward_bytes + 511) & ~511ull, footprints[1], nullptr, nullptr, nullptr);
    bytes = footprints[1][slices - 1].Offset + (UINT64)footprints[1][slices - 1].Footprint.RowPitch * depth_desc.Height;
    auto readback_heap = heap;
    readback_heap.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = bytes;
    buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = 1;
    buffer.SampleDesc.Count = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (!ok("Readback creation", dev->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &buffer,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback)))) return 2;

    ComPtr<ID3D12DescriptorHeap> dsv_heap, cpu_heap, gpu_heap;
    D3D12_DESCRIPTOR_HEAP_DESC dsv_heap_desc{ D3D12_DESCRIPTOR_HEAP_TYPE_DSV, slices };
    D3D12_DESCRIPTOR_HEAP_DESC cpu_heap_desc{ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, slices };
    auto gpu_heap_desc = cpu_heap_desc;
    gpu_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (!ok("DSV heap", dev->CreateDescriptorHeap(&dsv_heap_desc, IID_PPV_ARGS(&dsv_heap))) ||
        !ok("CPU view heap", dev->CreateDescriptorHeap(&cpu_heap_desc, IID_PPV_ARGS(&cpu_heap))) ||
        !ok("GPU view heap", dev->CreateDescriptorHeap(&gpu_heap_desc, IID_PPV_ARGS(&gpu_heap)))) return 2;

    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    if (!ok("Queue", dev->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue))) ||
        !ok("Allocator", dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))) ||
        !ok("List", dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list))) ||
        !ok("Fence", dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) return 2;

    auto transition = [&](ID3D12Resource* res, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Transition = {res, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
        list->ResourceBarrier(1, &barrier);
    };
    auto texture_location = [](ID3D12Resource* res, UINT slice) {
        D3D12_TEXTURE_COPY_LOCATION copy{};
        copy.pResource = res;
        copy.SubresourceIndex = slice;
        return copy;
    };
    auto copy_array = [&](ID3D12Resource* dst, ID3D12Resource* src) {
        for (UINT i = 0; i < slices; ++i) {
            const auto from = texture_location(src, i), to = texture_location(dst, i);
            list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        }
    };
    auto read_array = [&](ID3D12Resource* src, UINT phase) {
        for (UINT i = 0; i < slices; ++i) {
            const auto from = texture_location(src, i);
            D3D12_TEXTURE_COPY_LOCATION to{};
            to.pResource = readback.Get();
            to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            to.PlacedFootprint = footprints[phase][i];
            list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        }
    };

    auto dsv = dsv_heap->GetCPUDescriptorHandleForHeapStart();
    const auto dsv_stride = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
    if (shared) transition(depth.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_DEPTH_WRITE);
    for (UINT i = 0; i < slices; ++i) {
        D3D12_DEPTH_STENCIL_VIEW_DESC view{};
        view.Format = DXGI_FORMAT_D16_UNORM;
        view.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
        view.Texture2DArray = {0, i, 1};
        dev->CreateDepthStencilView(depth.Get(), &view, dsv);
        list->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, (i + 1) * 8192 / 65535.0f, 0, 0, nullptr);
        dsv.ptr += dsv_stride;
    }
    transition(depth.Get(), D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE);
    if (!shared) {
        copy_array(uav.Get(), depth.Get());
        transition(uav.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    }
    read_array(uav.Get(), 0);
    transition(uav.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    ID3D12DescriptorHeap* bound[]{gpu_heap.Get()};
    list->SetDescriptorHeaps(1, bound);
    auto cpu = cpu_heap->GetCPUDescriptorHandleForHeapStart();
    auto mirror = gpu_heap->GetCPUDescriptorHandleForHeapStart();
    auto gpu = gpu_heap->GetGPUDescriptorHandleForHeapStart();
    const auto uav_stride = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    for (UINT i = 0; i < slices; ++i) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
        view.Format = DXGI_FORMAT_R16_UNORM;
        view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        view.Texture2DArray = {0, i, 1, 0};
        dev->CreateUnorderedAccessView(uav.Get(), nullptr, &view, cpu);
        dev->CopyDescriptorsSimple(1, mirror, cpu, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        const float value[4]{(6 - i) * 8192 / 65535.0f};
        list->ClearUnorderedAccessViewFloat(gpu, cpu, uav.Get(), value, 0, nullptr);
        cpu.ptr += uav_stride;
        mirror.ptr += uav_stride;
        gpu.ptr += uav_stride;
    }
    if (shared) {
        transition(depth.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_DEPTH_WRITE);
        transition(depth.Get(), D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE);
    } else {
        transition(uav.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        transition(depth.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
        copy_array(depth.Get(), uav.Get());
        transition(depth.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    }
    read_array(depth.Get(), 1);

    if (!ok("Close", list->Close())) return 2;
    ID3D12CommandList* submitted[]{list.Get()};
    queue->ExecuteCommandLists(1, submitted);
    if (!ok("Signal", queue->Signal(fence.Get(), 1))) return 2;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) return 2;
    const auto notified = fence->SetEventOnCompletion(1, event);
    const DWORD waited = SUCCEEDED(notified) ? WaitForSingleObject(event, 10000) : WAIT_FAILED;
    CloseHandle(event);
    if (waited != WAIT_OBJECT_0 || !ok("Device state", dev->GetDeviceRemovedReason())) return 2;

    void* mapping{};
    if (!ok("Readback map", readback->Map(0, nullptr, &mapping))) return 2;
    UINT failures = 0;
    for (UINT phase = 0; phase < 2; ++phase) {
        for (UINT i = 0; i < slices; ++i) {
            const auto& layout = footprints[phase][i];
            const auto expected = (phase ? 6 - i : i + 1) * 8192;
            UINT mismatches = 0;
            for (UINT y = 0; y < depth_desc.Height; ++y) {
                const auto row = reinterpret_cast<const std::uint16_t*>(
                    reinterpret_cast<const char*>(mapping) + layout.Offset + y * layout.Footprint.RowPitch);
                for (UINT x = 0; x < depth_desc.Width; ++x) mismatches += row[x] != expected;
            }
            if (mismatches) {
                const auto* first = reinterpret_cast<const std::uint16_t*>(reinterpret_cast<const char*>(mapping) + layout.Offset);
                std::printf("Phase %u slice %u: expected %u, first texel %u, mismatches %u\n",
                    phase, i, expected, unsigned(*first), mismatches);
            }
            failures += mismatches;
        }
    }
    const D3D12_RANGE unmodified{};
    readback->Unmap(0, &unmodified);
    std::printf("%s depth <-> UAV, 6 slices each way: %u mismatched texels\n", shared ? "Shared" : "Split", failures);
    return failures ? 1 : 0;
}

int main(int argc, char** argv) {
    const bool warp = argc >= 2 && std::strcmp(argv[1], "preview-warp") == 0;
    const bool preview = warp || (argc >= 2 && std::strcmp(argv[1], "preview") == 0);
    const bool experimental = preview && argc == 3 && std::strcmp(argv[2], "experimental") == 0;
    const bool split_retail = argc == 2 && std::strcmp(argv[1], "split-retail") == 0;
    const bool retail = split_retail || (argc == 2 && std::strcmp(argv[1], "retail") == 0);
    const bool split = split_retail || (argc == 2 && std::strcmp(argv[1], "split") == 0);
    ComPtr<ID3D12DeviceFactory> factory;
    if (preview || retail) {
        const GUID sdk_configuration = {0x7cda6aca, 0xa03e, 0x49c8, {0x94, 0x58, 0x03, 0x34, 0xd2, 0x0e, 0x07, 0xce}};
        ComPtr<ID3D12SDKConfiguration1> sdk;
        auto get_interface = reinterpret_cast<decltype(&D3D12GetInterface)>(GetProcAddress(
            LoadLibraryExW(L"d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32), "D3D12GetInterface"));
        if (!get_interface) return 2;
        HRESULT hr = get_interface(sdk_configuration, IID_PPV_ARGS(&sdk));
        result("SDK configuration", hr);
        if (FAILED(hr)) return 2;
        char path[MAX_PATH]{};
        if (!GetModuleFileNameA(nullptr, path, MAX_PATH)) return 2;
        char* end = std::strrchr(path, '\\');
        if (!end) return 2;
        if (strcpy_s(end + 1, sizeof(path) - (end + 1 - path),
            preview ? "agility\\build\\native\\bin\\x64\\" : "retail\\build\\native\\bin\\x64\\") != 0) return 2;
        hr = sdk->CreateDeviceFactory(preview ? 722 : 620, path, IID_PPV_ARGS(&factory));
        result(preview ? "Agility 722 factory" : "Agility 620 factory", hr);
        if (FAILED(hr)) return 2;
        hr = factory->ApplyToGlobalState();
        result("Apply app-local Agility SDK", hr);
        if (FAILED(hr)) return 2;
        if (experimental) {
            const GUID shader_models = {0x76f5573e, 0xf13a, 0x40f5, {0xb2, 0x97, 0x81, 0xce, 0x9e, 0x18, 0x93, 0x3f}};
            hr = factory->EnableExperimentalFeatures(1, &shader_models, nullptr, nullptr);
            result("Experimental shader models", hr);
            if (FAILED(hr)) return 2;
        }
    }

    ComPtr<IDXGIFactory6> dxgi;
    if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&dxgi)))) return 2;
    ComPtr<IDXGIAdapter1> adapter;
    auto hr = warp ? dxgi->EnumWarpAdapter(IID_PPV_ARGS(&adapter))
                   : dxgi->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter));
    if (FAILED(hr)) { result("GPU adapter", hr); return 2; }
    DXGI_ADAPTER_DESC1 adapter_desc{};
    if (FAILED(adapter->GetDesc1(&adapter_desc))) return 2;
    std::printf("Adapter: %ls\n", adapter_desc.Description);

    if (split) {
        ComPtr<ID3D12Debug> debug;
        const GUID debug_clsid = {0xf2352aeb, 0xdd84, 0x49fe, {0xb9, 0x7b, 0xa9, 0xdc, 0xfd, 0xcc, 0x1b, 0x4f}};
        const auto hr_debug = factory ? factory->GetConfigurationInterface(debug_clsid, IID_PPV_ARGS(&debug))
                                      : D3D12GetDebugInterface(IID_PPV_ARGS(&debug));
        result("Debug layer", hr_debug);
        if (split_retail && FAILED(hr_debug)) return 2;
        if (SUCCEEDED(hr_debug)) debug->EnableDebugLayer();
    }
    ComPtr<ID3D12Device> dev;
    hr = factory ? factory->CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev))
                 : D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev));
    result("CreateDevice", hr);
    if (FAILED(hr)) return 2;
    wchar_t runtime[MAX_PATH]{};
    if (GetModuleFileNameW(GetModuleHandleW(L"D3D12Core.dll"), runtime, MAX_PATH)) {
        std::printf("D3D12Core: %ls\n", runtime);
    }
    if (warp && GetModuleFileNameW(GetModuleHandleW(L"d3d10warp.dll"), runtime, MAX_PATH)) {
        std::printf("WARP: %ls\n", runtime);
    }

    D3D12_FEATURE_DATA_D3D12_OPTIONS_PREVIEW options{};
    hr = dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS_PREVIEW, &options, sizeof(options));
    result("Preview option query", hr);
    std::printf("Depth UAV capability: %d\n", options.UAVOfDepthStencilSupported);
    D3D12_FEATURE_DATA_FORMAT_SUPPORT format{DXGI_FORMAT_R16_UNORM};
    hr = dev->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &format, sizeof(format));
    result("R16_UNORM format query", hr);
    std::printf("R16_UNORM typed UAV load/store: %d/%d\n",
        bool(format.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD),
        bool(format.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE));

    D3D12_RESOURCE_DESC tex{};
    tex.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    tex.Width = tex.Height = 768;
    tex.DepthOrArraySize = 6;
    tex.MipLevels = 1;
    tex.Format = DXGI_FORMAT_R16_TYPELESS;
    tex.SampleDesc.Count = 1;
    tex.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    const auto alloc = dev->GetResourceAllocationInfo(0, 1, &tex);
    std::printf("Legacy allocation size: %llu\n", static_cast<unsigned long long>(alloc.SizeInBytes));
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap.CreationNodeMask = heap.VisibleNodeMask = 1;
    if (split) {
        ComPtr<ID3D12InfoQueue> info;
        dev.As(&info);
        if (info) info->ClearStoredMessages();
        const auto status = coherence_test(dev.Get(), tex, heap, false);
        if (!info) return split_retail ? 2 : status;
        UINT errors = 0;
        for (UINT64 i = 0; i < info->GetNumStoredMessages(); ++i) {
            SIZE_T size{};
            if (FAILED(info->GetMessage(i, nullptr, &size))) return 2;
            auto msg = static_cast<D3D12_MESSAGE*>(HeapAlloc(GetProcessHeap(), 0, size));
            if (!msg) return 2;
            if (SUCCEEDED(info->GetMessage(i, msg, &size)) && msg->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
                std::printf("D3D12 error: %s\n", msg->pDescription);
                ++errors;
            }
            HeapFree(GetProcessHeap(), 0, msg);
        }
        std::printf("Debug layer errors: %u\n", errors);
        return errors ? 1 : status;
    }
    D3D12_CLEAR_VALUE clear{};
    clear.Format = DXGI_FORMAT_D16_UNORM;
    ComPtr<ID3D12Resource> legacy;
    hr = dev->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &tex,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&legacy));
    result("Legacy UAV, no clear", hr);
    hr = dev->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &tex,
        D3D12_RESOURCE_STATE_COMMON, &clear, IID_PPV_ARGS(&legacy));
    result("Legacy COMMON, depth clear", hr);

    ComPtr<ID3D12Device10> modern;
    hr = dev.As(&modern);
    result("Enhanced creation interface", hr);
    if (FAILED(hr)) return 2;
    D3D12_RESOURCE_DESC1 tex1{};
    tex1.Dimension = tex.Dimension;
    tex1.Width = tex.Width;
    tex1.Height = tex.Height;
    tex1.DepthOrArraySize = tex.DepthOrArraySize;
    tex1.MipLevels = tex.MipLevels;
    tex1.Format = tex.Format;
    tex1.SampleDesc = tex.SampleDesc;
    tex1.Flags = tex.Flags;
    ComPtr<ID3D12Resource> enhanced;
    hr = modern->CreateCommittedResource3(&heap, D3D12_HEAP_FLAG_NONE, &tex1,
        D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS, nullptr, nullptr, 0, nullptr, IID_PPV_ARGS(&enhanced));
    result("Enhanced UAV, no clear", hr);
    hr = modern->CreateCommittedResource3(&heap, D3D12_HEAP_FLAG_NONE, &tex1,
        D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS, &clear, nullptr, 0, nullptr, IID_PPV_ARGS(&enhanced));
    result("Enhanced UAV, depth clear", hr);
    if (options.UAVOfDepthStencilSupported) return coherence_test(dev.Get(), tex, heap, true);
    return FAILED(hr) && !legacy && !enhanced ? 1 : 0;
}
