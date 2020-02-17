#define _WIN32_WINNT 0x600
#include <stdio.h>

#include <d3d11.h>
#include <d3dcompiler.h>

#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")

HRESULT CompileComputeShader(_In_ LPCWSTR srcFile, _In_ LPCSTR entryPoint,
  _In_ ID3D11Device* device, _Outptr_ ID3DBlob** blob)
{
  if (!srcFile || !entryPoint || !device || !blob)
    return E_INVALIDARG;

  *blob = nullptr;

  UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#if defined( DEBUG ) || defined( _DEBUG )
  flags |= D3DCOMPILE_DEBUG;
#endif

  // We generally prefer to use the higher CS shader profile when possible as CS 5.0 is better performance on 11-class hardware
  LPCSTR profile = (device->GetFeatureLevel() >= D3D_FEATURE_LEVEL_11_0) ? "cs_5_0" : "cs_4_0";

  const D3D_SHADER_MACRO defines[] =
  {
      "EXAMPLE_DEFINE", "1",
      NULL, NULL
  };

  ID3DBlob* shaderBlob = nullptr;
  ID3DBlob* errorBlob = nullptr;
  HRESULT hr = D3DCompileFromFile(srcFile, defines, D3D_COMPILE_STANDARD_FILE_INCLUDE,
    entryPoint, profile,
    flags, 0, &shaderBlob, &errorBlob);
  if (FAILED(hr))
  {
    if (errorBlob)
    {
      OutputDebugStringA((char*)errorBlob->GetBufferPointer());
      errorBlob->Release();
    }

    if (shaderBlob)
      shaderBlob->Release();

    return hr;
  }

  *blob = shaderBlob;

  return hr;
}

int main()
{
  // Create Device
  const D3D_FEATURE_LEVEL lvl[] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0,
                                    D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0 };

  UINT createDeviceFlags = 0;
#ifdef _DEBUG
  createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

  ID3D11Device* device = nullptr;
  HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, createDeviceFlags, lvl, _countof(lvl),
    D3D11_SDK_VERSION, &device, nullptr, nullptr);
  if (hr == E_INVALIDARG)
  {
    // DirectX 11.0 Runtime doesn't recognize D3D_FEATURE_LEVEL_11_1 as a valid value
    hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, &lvl[1], _countof(lvl) - 1,
      D3D11_SDK_VERSION, &device, nullptr, nullptr);
  }

  if (FAILED(hr))
  {
    printf("Failed creating Direct3D 11 device %08X\n", hr);
    return -1;
  }

  // Verify compute shader is supported
  if (device->GetFeatureLevel() < D3D_FEATURE_LEVEL_11_0)
  {
    D3D11_FEATURE_DATA_D3D10_X_HARDWARE_OPTIONS hwopts = { 0 };
    (void)device->CheckFeatureSupport(D3D11_FEATURE_D3D10_X_HARDWARE_OPTIONS, &hwopts, sizeof(hwopts));
    if (!hwopts.ComputeShaders_Plus_RawAndStructuredBuffers_Via_Shader_4_x)
    {
      device->Release();
      printf("DirectCompute is not supported by this device\n");
      return -1;
    }
  }

  // Compile shader
  ID3DBlob* csBlob = nullptr;
   hr = CompileComputeShader(L"ExampleCompute.hlsl", "CSMain", device, &csBlob);
  if (FAILED(hr))
  {
    device->Release();
    printf("Failed compiling shader %08X\n", hr);
    return -1;
  }

  // Create shader
  ID3D11ComputeShader* computeShader = nullptr;
  hr = device->CreateComputeShader(csBlob->GetBufferPointer(), csBlob->GetBufferSize(), nullptr, &computeShader);

  csBlob->Release();

  if (FAILED(hr))
  {
    device->Release();
  }

  printf("Success\n");

  // Clean up
  computeShader->Release();

  device->Release();

  return 0;
}