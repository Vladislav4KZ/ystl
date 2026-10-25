// sse packed math helpers derived from the cephes library

#pragma once

// this header is included inside namespace ystl::simd

// float constants
inline constexpr float k4Pi = 1.27323954473516f;
inline constexpr float kMinusDp1 = -0.78515625f;
inline constexpr float kMinusDp2 = -2.4187564849853515625e-4f;
inline constexpr float kMinusDp3 = -3.77489497744594108e-8f;
inline constexpr float kSinP0 = -1.9515295891e-4f;
inline constexpr float kSinP1 = 8.3321608736e-3f;
inline constexpr float kSinP2 = -1.6666654611e-1f;
inline constexpr float kCosP0 = 2.443315711809948e-5f;
inline constexpr float kCosP1 = -1.388731625493765e-3f;
inline constexpr float kCosP2 = 4.166664568298827e-2f;
inline constexpr float kPi = 3.14159265358979323846f;
inline constexpr float kPi2 = 1.57079632679489661923f;
inline constexpr float kPi4 = 0.78539816339744830962f;
inline constexpr float kAtanQ0 = 2.414213562373095f;
inline constexpr float kAtanQ1 = 0.414213562373095f;
inline constexpr float kAtanP0 = 8.05374449538e-2f;
inline constexpr float kAtanP1 = -1.38776856032e-1f;
inline constexpr float kAtanP2 = 1.99777106478e-1f;
inline constexpr float kAtanP3 = -3.33329491539e-1f;
inline constexpr float kSqrt1_2 = 0.70710678118654752440f;
inline constexpr float kLog2E = 1.44269504088896340736f;
inline constexpr float kLog2EA = 0.44269504088896340735992f;
inline constexpr float kLog210 = 3.32192809488736234787f;
inline constexpr float kLog102A = 3.0078125e-1f;
inline constexpr float kLog102B = 2.48745663981195213739e-4f;
inline constexpr float kLog10EA = 4.3359375e-1f;
inline constexpr float kLog10EB = 7.00731903251827651129e-4f;
inline constexpr float kLogP0 = 7.0376836292e-2f;
inline constexpr float kLogP1 = -1.1514610310e-1f;
inline constexpr float kLogP2 = 1.1676998740e-1f;
inline constexpr float kLogP3 = -1.2420140846e-1f;
inline constexpr float kLogP4 = 1.4249322787e-1f;
inline constexpr float kLogP5 = -1.6668057665e-1f;
inline constexpr float kLogP6 = 2.0000714765e-1f;
inline constexpr float kLogP7 = -2.4999993993e-1f;
inline constexpr float kLogP8 = 3.3333331174e-1f;
inline constexpr float kLogQ1 = -2.12194440e-4f;
inline constexpr float kLogQ2 = 0.693359375f;
inline constexpr float kExpC1 = 0.693359375f;
inline constexpr float kExpC2 = -2.12194440e-4f;
inline constexpr float kExpP0 = 1.9875691500e-4f;
inline constexpr float kExpP1 = 1.3981999507e-3f;
inline constexpr float kExpP2 = 8.3334519073e-3f;
inline constexpr float kExpP3 = 4.1665795894e-2f;
inline constexpr float kExpP4 = 1.6666665459e-1f;
inline constexpr float kExpP5 = 5.0000001201e-1f;
inline constexpr float kExp2P0 = 1.535336188319500e-4f;
inline constexpr float kExp2P1 = 1.339887440266574e-3f;
inline constexpr float kExp2P2 = 9.618437357674640e-3f;
inline constexpr float kExp2P3 = 5.550332471162809e-2f;
inline constexpr float kExp2P4 = 2.402264791363012e-1f;
inline constexpr float kExp2P5 = 6.931472028550421e-1f;
inline constexpr float kExp10P0 = 2.063216740311022e-1f;
inline constexpr float kExp10P1 = 5.420251702225484e-1f;
inline constexpr float kExp10P2 = 1.171292686296281f;
inline constexpr float kExp10P3 = 2.034649854009453f;
inline constexpr float kExp10P4 = 2.650948748208892f;
inline constexpr float kExp10P5 = 2.302585167056758f;

// saturation ranges that keep the power-of-two scaling inside float exponents
inline constexpr float kExpMax = 88.7f;
inline constexpr float kExpMin = -87.0f;
inline constexpr float kExp2Max = 128.0f;
inline constexpr float kExp2Min = -126.0f;
inline constexpr float kExp10Max = 38.6f;
inline constexpr float kExp10Min = -37.0f;

// integer bit patterns
inline constexpr int32_t kSignBit = static_cast<int32_t> (0x80000000u);
inline constexpr int32_t kInvMantBits = static_cast<int32_t> (~0x7f800000);
inline constexpr int32_t kInfBits = 0x7f800000;
inline constexpr int32_t kMaxBits = 0x7f7fffff;

[[nodiscard]] YSTL_FORCE_INLINE __m128 set1 (const float value) noexcept {
  return _mm_set1_ps (value);
}

[[nodiscard]] YSTL_FORCE_INLINE __m128i set1i (const int32_t value) noexcept {
  return _mm_set1_epi32 (value);
}

[[nodiscard]] YSTL_FORCE_INLINE __m128 sign_mask () noexcept {
  return _mm_castsi128_ps (set1i (kSignBit));
}

[[nodiscard]] YSTL_FORCE_INLINE __m128 sign_bits (const __m128 &value) noexcept {
  return _mm_and_ps (value, sign_mask ());
}

[[nodiscard]] YSTL_FORCE_INLINE __m128 fabs_ps (const __m128 &value) noexcept {
  return _mm_andnot_ps (sign_mask (), value);
}

YSTL_FORCE_INLINE void sincos_ps (const __m128 &ang, __m128 &rSin, __m128 &rCos) noexcept {
  __m128 x = fabs_ps (ang);
  __m128 tmp = _mm_mul_ps (x, set1 (k4Pi));

  __m128i j = _mm_cvttps_epi32 (tmp);
  j = _mm_add_epi32 (j, set1i (1));
  j = _mm_and_si128 (j, set1i (~1));
  tmp = _mm_cvtepi32_ps (j);

  x = _mm_add_ps (x, _mm_mul_ps (tmp, set1 (kMinusDp1)));
  x = _mm_add_ps (x, _mm_mul_ps (tmp, set1 (kMinusDp2)));
  x = _mm_add_ps (x, _mm_mul_ps (tmp, set1 (kMinusDp3)));

  __m128i j2 = _mm_sub_epi32 (j, set1i (2));
  __m128 cos_sign = _mm_castsi128_ps (_mm_slli_epi32 (_mm_andnot_si128 (j2, set1i (4)), 29));
  __m128 sin_sign = _mm_xor_ps (sign_bits (ang), _mm_castsi128_ps (_mm_slli_epi32 (_mm_and_si128 (j, set1i (4)), 29)));

  __m128 xx = _mm_mul_ps (x, x);

  __m128 cos_poly = set1 (kCosP0);
  cos_poly = _mm_mul_ps (cos_poly, xx);
  cos_poly = _mm_add_ps (cos_poly, set1 (kCosP1));
  cos_poly = _mm_mul_ps (cos_poly, xx);
  cos_poly = _mm_add_ps (cos_poly, set1 (kCosP2));
  cos_poly = _mm_mul_ps (cos_poly, xx);
  cos_poly = _mm_mul_ps (cos_poly, xx);
  cos_poly = _mm_sub_ps (cos_poly, _mm_mul_ps (xx, set1 (0.5f)));
  cos_poly = _mm_add_ps (cos_poly, set1 (1.0f));

  __m128 sin_poly = set1 (kSinP0);
  sin_poly = _mm_mul_ps (sin_poly, xx);
  sin_poly = _mm_add_ps (sin_poly, set1 (kSinP1));
  sin_poly = _mm_mul_ps (sin_poly, xx);
  sin_poly = _mm_add_ps (sin_poly, set1 (kSinP2));
  sin_poly = _mm_mul_ps (sin_poly, xx);
  sin_poly = _mm_mul_ps (sin_poly, x);
  sin_poly = _mm_add_ps (sin_poly, x);

  j = _mm_and_si128 (j, set1i (2));
  j = _mm_cmpeq_epi32 (j, _mm_setzero_si128 ());

  __m128 poly_mask = _mm_castsi128_ps (j);
  __m128 sin2 = _mm_and_ps (poly_mask, sin_poly);
  __m128 sin1 = _mm_andnot_ps (poly_mask, cos_poly);

  sin_poly = _mm_sub_ps (sin_poly, sin2);
  cos_poly = _mm_sub_ps (cos_poly, sin1);

  rSin = _mm_xor_ps (_mm_add_ps (sin1, sin2), sin_sign);
  rCos = _mm_xor_ps (_mm_add_ps (cos_poly, sin_poly), cos_sign);
}

[[nodiscard]] YSTL_FORCE_INLINE __m128 sin_ps (const __m128 &ang) noexcept {
  __m128 s {};
  __m128 c {};

  sincos_ps (ang, s, c);
  return s;
}

[[nodiscard]] YSTL_FORCE_INLINE __m128 cos_ps (const __m128 &ang) noexcept {
  __m128 s {};
  __m128 c {};

  sincos_ps (ang, s, c);
  return c;
}

[[nodiscard]] YSTL_FORCE_INLINE __m128 tan_ps (const __m128 &ang) noexcept {
  __m128 s {};
  __m128 c {};

  sincos_ps (ang, s, c);
  return _mm_div_ps (s, c);
}

[[nodiscard]] YSTL_FORCE_INLINE __m128 atan_ps (const __m128 &val) noexcept {
  __m128 x = fabs_ps (val);

  __m128 mask1 = _mm_cmplt_ps (set1 (kAtanQ0), x);
  __m128 mask2 = _mm_andnot_ps (mask1, _mm_cmplt_ps (set1 (kAtanQ1), x));
  __m128 mask3 = _mm_or_ps (mask1, mask2);

  __m128 y = _mm_and_ps (mask1, set1 (kPi2));
  y = _mm_add_ps (y, _mm_and_ps (mask2, set1 (kPi4)));

  __m128 x1 = _mm_div_ps (set1 (-1.0f), x);
  __m128 x2 = _mm_div_ps (_mm_sub_ps (x, set1 (1.0f)), _mm_add_ps (x, set1 (1.0f)));

  x = _mm_andnot_ps (mask3, x);
  x = _mm_add_ps (x, _mm_and_ps (mask1, x1));
  x = _mm_add_ps (x, _mm_and_ps (mask2, x2));

  __m128 z = _mm_mul_ps (x, x);
  __m128 tmp = _mm_mul_ps (set1 (kAtanP0), z);
  tmp = _mm_mul_ps (_mm_add_ps (tmp, set1 (kAtanP1)), z);
  tmp = _mm_mul_ps (_mm_add_ps (tmp, set1 (kAtanP2)), z);
  tmp = _mm_mul_ps (_mm_add_ps (tmp, set1 (kAtanP3)), z);
  tmp = _mm_add_ps (_mm_mul_ps (tmp, x), _mm_add_ps (x, y));

  return _mm_xor_ps (tmp, sign_bits (val));
}

[[nodiscard]] YSTL_FORCE_INLINE __m128 atan2_ps (const __m128 &y, const __m128 &x) noexcept {
  __m128 ygt0 = _mm_cmplt_ps (_mm_setzero_ps (), y);
  __m128 ylt0 = _mm_cmplt_ps (y, _mm_setzero_ps ());

  __m128 tmp1 = _mm_and_ps (ygt0, set1 (kPi2));
  __m128 tmp2 = _mm_and_ps (ylt0, set1 (kPi2));
  __m128 val = _mm_sub_ps (tmp1, tmp2);

  __m128 xlt0 = _mm_cmplt_ps (x, _mm_setzero_ps ());

  ygt0 = _mm_andnot_ps (ylt0, xlt0);
  ylt0 = _mm_and_ps (ylt0, xlt0);
  tmp1 = _mm_and_ps (ygt0, set1 (kPi));
  tmp2 = _mm_and_ps (ylt0, set1 (kPi));
  __m128 offs = _mm_sub_ps (tmp1, tmp2);

  __m128 xeq0 = _mm_cmpeq_ps (x, _mm_setzero_ps ());
  __m128 atan = _mm_add_ps (atan_ps (_mm_div_ps (y, x)), offs);

  return _mm_add_ps (_mm_andnot_ps (xeq0, atan), _mm_and_ps (xeq0, val));
}

[[nodiscard]] YSTL_FORCE_INLINE __m128 ceil_ps (const __m128 &val) noexcept {
  __m128i trunc = _mm_cvttps_epi32 (val);
  __m128 mask = _mm_castsi128_ps (_mm_cmpeq_epi32 (trunc, set1i (kSignBit)));
  mask = _mm_or_ps (mask, _mm_cmpeq_ps (_mm_cvtepi32_ps (trunc), val));

  __m128i sign_off = _mm_xor_si128 (set1i (1), _mm_srli_epi32 (_mm_castps_si128 (val), 31));
  trunc = _mm_add_epi32 (trunc, sign_off);

  __m128 x = _mm_cvtepi32_ps (trunc);
  x = _mm_andnot_ps (mask, x);
  return _mm_add_ps (x, _mm_and_ps (mask, val));
}

[[nodiscard]] YSTL_FORCE_INLINE __m128 floor_ps (const __m128 &val) noexcept {
  __m128i trunc = _mm_cvttps_epi32 (val);
  __m128 mask = _mm_castsi128_ps (_mm_cmpeq_epi32 (trunc, set1i (kSignBit)));
  mask = _mm_or_ps (mask, _mm_cmpeq_ps (_mm_cvtepi32_ps (trunc), val));

  __m128i sign_off = _mm_srli_epi32 (_mm_castps_si128 (val), 31);
  trunc = _mm_sub_epi32 (trunc, sign_off);

  __m128 x = _mm_cvtepi32_ps (trunc);
  x = _mm_andnot_ps (mask, x);
  return _mm_add_ps (x, _mm_and_ps (mask, val));
}

[[nodiscard]] YSTL_FORCE_INLINE __m128 trunc_ps (const __m128 &val) noexcept {
  __m128i trunc = _mm_cvttps_epi32 (val);
  __m128 mask = _mm_castsi128_ps (_mm_cmpeq_epi32 (trunc, set1i (kSignBit)));

  __m128 x = _mm_cvtepi32_ps (trunc);
  x = _mm_andnot_ps (mask, x);
  return _mm_add_ps (x, _mm_and_ps (mask, val));
}

[[nodiscard]] YSTL_FORCE_INLINE __m128 modf_ps (const __m128 &val, __m128 &rInt) noexcept {
  rInt = trunc_ps (val);
  return _mm_sub_ps (val, rInt);
}

[[nodiscard]] YSTL_FORCE_INLINE __m128 round_ps (const __m128 &val) noexcept {
  // halve away from zero, like std::round: trunc ( val + copysign ( 0.5, val ) )
  __m128 half = _mm_or_ps (set1 (0.5f), sign_bits (val));
  return trunc_ps (_mm_add_ps (val, half));
}

struct LogParts {
  __m128 x {}; // reduced fraction
  __m128 poly {}; // polynomial * x * x^2
  __m128 exp {}; // exponent as float
  __m128 xx {}; // x^2
  __m128 invalid {}; // val <= 0
};

[[nodiscard]] YSTL_FORCE_INLINE LogParts log_parts (const __m128 &val) noexcept {
  LogParts p {};

  __m128i e = _mm_srli_epi32 (_mm_castps_si128 (val), 23);
  e = _mm_sub_epi32 (e, set1i (0x7f));
  p.exp = _mm_add_ps (_mm_cvtepi32_ps (e), set1 (1.0f));

  p.x = _mm_or_ps (_mm_and_ps (val, _mm_castsi128_ps (set1i (kInvMantBits))), set1 (0.5f));

  __m128 mask = _mm_cmplt_ps (p.x, set1 (kSqrt1_2));
  __m128 tmp = _mm_and_ps (p.x, mask);

  p.x = _mm_sub_ps (p.x, set1 (1.0f));
  p.exp = _mm_sub_ps (p.exp, _mm_and_ps (set1 (1.0f), mask));
  p.x = _mm_add_ps (p.x, tmp);
  p.invalid = _mm_cmple_ps (val, _mm_setzero_ps ());

  p.xx = _mm_mul_ps (p.x, p.x);

  __m128 poly = set1 (kLogP0);
  poly = _mm_add_ps (_mm_mul_ps (poly, p.x), set1 (kLogP1));
  poly = _mm_add_ps (_mm_mul_ps (poly, p.x), set1 (kLogP2));
  poly = _mm_add_ps (_mm_mul_ps (poly, p.x), set1 (kLogP3));
  poly = _mm_add_ps (_mm_mul_ps (poly, p.x), set1 (kLogP4));
  poly = _mm_add_ps (_mm_mul_ps (poly, p.x), set1 (kLogP5));
  poly = _mm_add_ps (_mm_mul_ps (poly, p.x), set1 (kLogP6));
  poly = _mm_add_ps (_mm_mul_ps (poly, p.x), set1 (kLogP7));
  poly = _mm_add_ps (_mm_mul_ps (poly, p.x), set1 (kLogP8));
  p.poly = _mm_mul_ps (_mm_mul_ps (poly, p.x), p.xx);

  return p;
}

[[nodiscard]] YSTL_FORCE_INLINE __m128 log_ps (const __m128 &val) noexcept {
  auto p = log_parts (val);

  __m128 y = _mm_add_ps (p.poly, _mm_mul_ps (p.exp, set1 (kLogQ1)));
  y = _mm_sub_ps (y, _mm_mul_ps (p.xx, set1 (0.5f)));

  __m128 x = _mm_add_ps (p.x, _mm_mul_ps (p.exp, set1 (kLogQ2)));
  x = _mm_add_ps (x, y);

  return _mm_or_ps (x, p.invalid);
}

[[nodiscard]] YSTL_FORCE_INLINE __m128 log2_ps (const __m128 &val) noexcept {
  auto p = log_parts (val);

  __m128 y = _mm_sub_ps (p.poly, _mm_mul_ps (p.xx, set1 (0.5f)));
  __m128 z = _mm_mul_ps (p.x, set1 (kLog2EA));

  z = _mm_add_ps (z, y);
  z = _mm_add_ps (z, _mm_mul_ps (y, set1 (kLog2EA)));
  z = _mm_add_ps (z, p.x);
  z = _mm_add_ps (z, p.exp);

  return _mm_or_ps (z, p.invalid);
}

[[nodiscard]] YSTL_FORCE_INLINE __m128 log10_ps (const __m128 &val) noexcept {
  auto p = log_parts (val);

  __m128 y = _mm_sub_ps (p.poly, _mm_mul_ps (p.xx, set1 (0.5f)));
  __m128 eb = _mm_mul_ps (p.exp, set1 (kLog102B));
  __m128 ea = _mm_mul_ps (p.exp, set1 (kLog102A));

  __m128 z = _mm_mul_ps (_mm_add_ps (p.x, y), set1 (kLog10EB));
  z = _mm_add_ps (z, _mm_mul_ps (y, set1 (kLog10EA)));
  z = _mm_add_ps (z, eb);
  z = _mm_add_ps (z, _mm_mul_ps (p.x, set1 (kLog10EA)));
  z = _mm_add_ps (z, ea);

  return _mm_or_ps (z, p.invalid);
}

// 2^e for an integer exponent lane, callers keep e in [-126, 128]
[[nodiscard]] YSTL_FORCE_INLINE __m128 scale_exp2 (const __m128i &e) noexcept {
  return _mm_castsi128_ps (_mm_slli_epi32 (_mm_add_epi32 (e, set1i (0x7f)), 23));
}

[[nodiscard]] YSTL_FORCE_INLINE __m128 exp2_ps (__m128 x) noexcept {
  // min/max with the constant first keeps nan in x, then saturates the exponent
  x = _mm_min_ps (set1 (kExp2Max), x);
  x = _mm_max_ps (set1 (kExp2Min), x);

  __m128i e = _mm_cvttps_epi32 (x);
  e = _mm_sub_epi32 (e, _mm_srli_epi32 (_mm_castps_si128 (x), 31));
  x = _mm_sub_ps (x, _mm_cvtepi32_ps (e));

  __m128i adjust = _mm_and_si128 (_mm_castps_si128 (_mm_cmplt_ps (set1 (0.5f), x)), set1i (1));
  x = _mm_sub_ps (x, _mm_cvtepi32_ps (adjust));
  e = _mm_add_epi32 (e, adjust);

  __m128 y = set1 (kExp2P0);
  y = _mm_add_ps (_mm_mul_ps (y, x), set1 (kExp2P1));
  y = _mm_add_ps (_mm_mul_ps (y, x), set1 (kExp2P2));
  y = _mm_add_ps (_mm_mul_ps (y, x), set1 (kExp2P3));
  y = _mm_add_ps (_mm_mul_ps (y, x), set1 (kExp2P4));
  y = _mm_add_ps (_mm_mul_ps (y, x), set1 (kExp2P5));
  y = _mm_add_ps (_mm_mul_ps (y, x), set1 (1.0f));

  y = _mm_mul_ps (y, scale_exp2 (e));
  return _mm_min_ps (y, _mm_castsi128_ps (set1i (kMaxBits)));
}

[[nodiscard]] YSTL_FORCE_INLINE __m128 exp_ps (const __m128 &val) noexcept {
  __m128 value = _mm_min_ps (set1 (kExpMax), val);
  value = _mm_max_ps (set1 (kExpMin), value);

  __m128 fx = _mm_add_ps (_mm_mul_ps (value, set1 (kLog2E)), set1 (0.5f));

  const __m128i whole = _mm_cvttps_epi32 (fx);
  __m128 tmp = _mm_cvtepi32_ps (whole);

  __m128 mask = _mm_cmple_ps (fx, tmp);
  fx = _mm_sub_ps (tmp, _mm_and_ps (mask, set1 (1.0f)));

  __m128 x = _mm_sub_ps (value, _mm_mul_ps (fx, set1 (kExpC1)));
  x = _mm_sub_ps (x, _mm_mul_ps (fx, set1 (kExpC2)));

  __m128 xx = _mm_mul_ps (x, x);

  __m128 y = set1 (kExpP0);
  y = _mm_add_ps (_mm_mul_ps (y, x), set1 (kExpP1));
  y = _mm_add_ps (_mm_mul_ps (y, x), set1 (kExpP2));
  y = _mm_add_ps (_mm_mul_ps (y, x), set1 (kExpP3));
  y = _mm_add_ps (_mm_mul_ps (y, x), set1 (kExpP4));
  y = _mm_add_ps (_mm_mul_ps (y, x), set1 (kExpP5));
  y = _mm_add_ps (_mm_mul_ps (y, xx), x);
  y = _mm_add_ps (y, set1 (1.0f));

  y = _mm_mul_ps (y, scale_exp2 (_mm_cvttps_epi32 (fx)));

  return _mm_min_ps (y, _mm_castsi128_ps (set1i (kMaxBits)));
}

[[nodiscard]] YSTL_FORCE_INLINE __m128 exp10_ps (const __m128 &val) noexcept {
  __m128 value = _mm_min_ps (set1 (kExp10Max), val);
  value = _mm_max_ps (set1 (kExp10Min), value);

  __m128 px = _mm_mul_ps (value, set1 (kLog210));
  __m128 qx = _mm_add_ps (px, set1 (0.5f));

  __m128i e = _mm_cvttps_epi32 (qx);
  e = _mm_sub_epi32 (e, _mm_srli_epi32 (_mm_castps_si128 (qx), 31));
  qx = _mm_cvtepi32_ps (e);

  __m128 x = _mm_sub_ps (value, _mm_mul_ps (qx, set1 (kLog102A)));
  x = _mm_sub_ps (x, _mm_mul_ps (qx, set1 (kLog102B)));

  __m128 is_zero = _mm_cmpeq_ps (val, _mm_setzero_ps ());

  __m128 y = set1 (kExp10P0);
  y = _mm_add_ps (_mm_mul_ps (y, x), set1 (kExp10P1));
  y = _mm_add_ps (_mm_mul_ps (y, x), set1 (kExp10P2));
  y = _mm_add_ps (_mm_mul_ps (y, x), set1 (kExp10P3));
  y = _mm_add_ps (_mm_mul_ps (y, x), set1 (kExp10P4));
  y = _mm_add_ps (_mm_mul_ps (y, x), set1 (kExp10P5));
  y = _mm_add_ps (_mm_mul_ps (y, x), set1 (1.0f));

  y = _mm_mul_ps (y, scale_exp2 (e));
  y = _mm_min_ps (y, _mm_castsi128_ps (set1i (kMaxBits)));
  y = _mm_andnot_ps (is_zero, y);

  return _mm_add_ps (y, _mm_and_ps (is_zero, set1 (1.0f)));
}

[[nodiscard]] YSTL_FORCE_INLINE __m128 pow_ps (const __m128 &base, const __m128 &exp) noexcept {
  auto p = log_parts (base);

  __m128 ln = _mm_add_ps (p.x, _mm_mul_ps (p.exp, set1 (kLogQ2)));
  ln = _mm_add_ps (ln, p.poly);
  ln = _mm_add_ps (ln, _mm_mul_ps (p.exp, set1 (kLogQ1)));
  ln = _mm_sub_ps (ln, _mm_mul_ps (p.xx, set1 (0.5f)));

  __m128 y = exp2_ps (_mm_mul_ps (ln, _mm_mul_ps (exp, set1 (kLog2E))));

  __m128 base_zero = _mm_cmpeq_ps (base, _mm_setzero_ps ());
  __m128 exp_zero = _mm_cmpeq_ps (exp, _mm_setzero_ps ());
  __m128 inf_mask = _mm_and_ps (base_zero, _mm_cmplt_ps (exp, _mm_setzero_ps ()));
  __m128 zero_mask = _mm_or_ps (base_zero, exp_zero);

  y = _mm_andnot_ps (zero_mask, y);
  y = _mm_add_ps (y, _mm_and_ps (exp_zero, set1 (1.0f)));
  return _mm_add_ps (y, _mm_and_ps (inf_mask, _mm_castsi128_ps (set1i (kInfBits))));
}

// ---- string kernels ----
// bounded scans never read outside [src, src + count), unbounded ones use
// aligned loads that stay inside the page; callers gate those on YSTL_ASAN.

[[nodiscard]] YSTL_FORCE_INLINE unsigned zero_bytes (const __m128i &value) noexcept {
  return static_cast<unsigned> (_mm_movemask_epi8 (_mm_cmpeq_epi8 (value, _mm_setzero_si128 ())));
}

// true when a 16-byte load at ptr would cross the 4k page boundary
[[nodiscard]] YSTL_FORCE_INLINE bool crosses_page (const void *ptr) noexcept {
  return (reinterpret_cast<uintptr_t> (ptr) & 0xFFFu) > 0x1000u - 16u;
}

// like memchr, but scans 16 bytes at a time while count allows
[[nodiscard]] YSTL_FORCE_INLINE const void *memchr_ps (const void *src, const uint8_t value, size_t count) noexcept {
  auto *bytes = static_cast<const char *> (src);
  const auto needle = _mm_set1_epi8 (static_cast<char> (value));

  while (count >= 16) {
    const auto mask =
      static_cast<unsigned> (_mm_movemask_epi8 (_mm_cmpeq_epi8 (_mm_loadu_si128 (reinterpret_cast<const __m128i *> (bytes)), needle)));

    if (mask) {
      return bytes + std::countr_zero (mask);
    }
    bytes += 16;
    count -= 16;
  }

  for (size_t i = 0; i < count; ++i) {
    if (static_cast<uint8_t> (bytes[i]) == value) {
      return bytes + i;
    }
  }
  return nullptr;
}

// strlen without a length bound, reads only aligned blocks of the same page
[[nodiscard]] YSTL_FORCE_INLINE size_t strlen_ps (const char *str) noexcept {
  const auto addr = reinterpret_cast<uintptr_t> (str);
  const auto offset = addr & 15u;
  auto *block = reinterpret_cast<const __m128i *> (addr & ~uintptr_t (15));

  auto mask = zero_bytes (_mm_load_si128 (block)) & (~0u << offset);
  if (mask) {
    return static_cast<size_t> (std::countr_zero (mask)) - static_cast<size_t> (offset);
  }

  size_t length = 16 - offset;
  ++block;
  while (true) {
    const auto wide = zero_bytes (_mm_load_si128 (block));

    if (wide) {
      return length + std::countr_zero (wide);
    }
    ++block;
    length += 16;
  }
}

// first occurrence of ch, or nullptr when the terminator comes first
[[nodiscard]] YSTL_FORCE_INLINE const char *strchr_ps (const char *str, const char ch) noexcept {
  const auto value = _mm_set1_epi8 (ch);
  const auto addr = reinterpret_cast<uintptr_t> (str);
  auto offset = addr & 15u;
  auto *block = reinterpret_cast<const __m128i *> (addr & ~uintptr_t (15));

  auto base = -static_cast<ptrdiff_t> (offset);
  while (true) {
    const auto chunk = _mm_load_si128 (block);
    auto hit = static_cast<unsigned> (_mm_movemask_epi8 (_mm_cmpeq_epi8 (chunk, value)));
    auto end = zero_bytes (chunk);

    if (offset) {
      hit &= ~0u << offset;
      end &= ~0u << offset;
      offset = 0;
    }

    if (hit || end) {
      if (!hit || (end && std::countr_zero (end) < std::countr_zero (hit))) {
        return nullptr;
      }
      return str + base + static_cast<ptrdiff_t> (std::countr_zero (hit));
    }
    ++block;
    base += 16;
  }
}

// strncmp over at most count bytes, stops on the first difference or terminator
[[nodiscard]] YSTL_FORCE_INLINE int strncmp_ps (const char *lhs, const char *rhs, size_t count) noexcept {
  while (count >= 16 && !crosses_page (lhs) && !crosses_page (rhs)) {
    const auto left = _mm_loadu_si128 (reinterpret_cast<const __m128i *> (lhs));
    const auto right = _mm_loadu_si128 (reinterpret_cast<const __m128i *> (rhs));

    const auto diff = ~static_cast<unsigned> (_mm_movemask_epi8 (_mm_cmpeq_epi8 (left, right))) & 0xFFFFu;
    const auto stop = diff | zero_bytes (left);

    if (stop) {
      const auto at = static_cast<size_t> (std::countr_zero (stop));
      return static_cast<int> (static_cast<uint8_t> (lhs[at])) - static_cast<int> (static_cast<uint8_t> (rhs[at]));
    }
    lhs += 16;
    rhs += 16;
    count -= 16;
  }

  for (size_t i = 0; i < count; ++i) {
    const auto left = static_cast<uint8_t> (lhs[i]);
    const auto right = static_cast<uint8_t> (rhs[i]);

    if (left != right) {
      return static_cast<int> (left) - static_cast<int> (right);
    }
    if (!left) {
      return 0;
    }
  }
  return 0;
}
