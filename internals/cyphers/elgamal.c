#include "elgamal.h"
#include "string.h"

#if defined(__SIZEOF_INT128__)
typedef unsigned __int128 u128;
#else
#error "__int128 required"
#endif

static u128 g_last_a = 0;
static int g_last_a_valid = 0;

static int s_len_local(const char* s) {
    int n = 0;
    if (!s) return 0;
    while (s[n]) ++n;
    return n;
}

static u128 parse_u128_slice(const char* s, int start, int end) {
    u128 v = 0;
    for (int i = start; i < end; ++i) {
        char c = s[i];
        if (c >= '0' && c <= '9') {
            v = v * (u128)10 + (u128)(c - '0');
        }
    }
    return v;
}

static int parse_block_u128(const char* s, int start, int end, u128* out, int cap) {
    int n = 0;
    int i = start;
    while (i < end && n < cap) {
        while (i < end && !(s[i] >= '0' && s[i] <= '9')) ++i;
        if (i >= end) break;
        int j = i;
        while (j < end && (s[j] >= '0' && s[j] <= '9')) ++j;
        out[n++] = parse_u128_slice(s, i, j);
        i = j;
    }
    return n;
}

static int parse_blocks(const char* s, int idx[3][2]) {
    int n = s_len_local(s);
    int count = 0;
    int i = 0;
    while (i < n && count < 3) {
        while (i < n && s[i] != '[') ++i;
        if (i >= n) break;
        int start = i + 1;
        int depth = 1;
        ++i;
        while (i < n && depth > 0) {
            if (s[i] == '[') depth++;
            else if (s[i] == ']') depth--;
            ++i;
        }
        int end = (depth == 0) ? (i - 1) : n;
        idx[count][0] = start;
        idx[count][1] = end;
        count++;
    }
    return count;
}

static int parse_frag_local(const char* frag, int* out, int cap) {
    int n = 0;
    int i = 0;
    while (frag[i] && n < cap) {
        while (frag[i] && !(frag[i] >= '0' && frag[i] <= '9')) ++i;
        if (!frag[i]) break;
        int v = 0;
        while (frag[i] && frag[i] >= '0' && frag[i] <= '9') {
            v = v * 10 + (frag[i] - '0');
            ++i;
        }
        out[n++] = v;
    }
    return n;
}

static u128 mod_add(u128 a, u128 b, u128 m) {
    a %= m;
    b %= m;
    u128 s = a + b;
    if (s >= m) s -= m;
    return s;
}

static u128 mod_sub(u128 a, u128 b, u128 m) {
    a %= m;
    b %= m;
    if (a >= b) return a - b;
    return m - (b - a);
}

static u128 mod_mul(u128 a, u128 b, u128 m) {
    a %= m;
    b %= m;
    u128 r = 0;
    while (b) {
        if (b & 1) r = (r + a) % m;
        a = (a + a) % m;
        b >>= 1;
    }
    return r;
}

static u128 mod_pow(u128 base, u128 exp, u128 mod) {
    base %= mod;
    u128 r = 1 % mod;
    while (exp) {
        if (exp & 1) r = mod_mul(r, base, mod);
        base = mod_mul(base, base, mod);
        exp >>= 1;
    }
    return r;
}

static u128 gcd_u128(u128 a, u128 b) {
    while (b) {
        u128 r = a % b;
        a = b;
        b = r;
    }
    return a;
}

static u128 egcd_u128(u128 a, u128 b, __int128* x, __int128* y) {
    if (b == 0) {
        *x = 1;
        *y = 0;
        return a;
    }
    __int128 x1 = 0, y1 = 0;
    u128 g = egcd_u128(b, a % b, &x1, &y1);
    __int128 nx = y1;
    __int128 ny = x1 - (__int128)(a / b) * y1;
    *x = nx;
    *y = ny;
    return g;
}

static u128 mod_inv_u128(u128 a, u128 m) {
    if (m == 0) return 0;
    __int128 x = 0, y = 0;
    u128 g = egcd_u128(a % m, m, &x, &y);
    if (g != 1) return 0;
    __int128 mm = (__int128)m;
    __int128 r = x % mm;
    if (r < 0) r += mm;
    return (u128)r;
}

static void u128_to_dec(u128 x, char* buf, int cap) {
    char tmp[64];
    int n = 0;
    if (cap <= 0) return;
    if (x == 0) {
        if (cap > 1) {
            buf[0] = '0';
            buf[1] = 0;
        } else {
            buf[0] = 0;
        }
        return;
    }
    while (x && n < (int)sizeof(tmp)) {
        unsigned int d = (unsigned int)(x % 10);
        tmp[n++] = (char)('0' + d);
        x /= 10;
    }
    if (n + 1 > cap) {
        buf[0] = 0;
        return;
    }
    int k = 0;
    while (n > 0) {
        buf[k++] = tmp[--n];
    }
    buf[k] = 0;
}

static void num_to_text_utf8(const char* alph, u128 m, char* out, int cap) {
    if (!alph || !out || cap <= 0) {
        if (cap > 0) out[0] = 0;
        return;
    }
    uint32_t alph_cp[64];
    int al = utf8_to_u32(alph, alph_cp, 64);
    if (al <= 0) {
        out[0] = 0;
        return;
    }
    uint32_t msg_cp[256];
    int tn = 0;
    while (m && tn < 256) {
        u128 q = m / 100;
        u128 r = m % 100;
        int idx = (int)r - 1;
        uint32_t cp = '?';
        if (idx >= 0 && idx < al) cp = alph_cp[idx];
        msg_cp[tn++] = cp;
        m = q;
    }
    if (tn == 0) {
        out[0] = 0;
        return;
    }
    for (int i = 0; i < tn / 2; ++i) {
        uint32_t t = msg_cp[i];
        msg_cp[i] = msg_cp[tn - 1 - i];
        msg_cp[tn - 1 - i] = t;
    }
    u32_to_utf8(msg_cp, tn, out, cap);
}

static const char* task1_verify(const char* block) {
    static char out[32];
    u128 vals[16];
    int n = parse_block_u128(block, 0, s_len_local(block), vals, 16);
    if (n < 10) {
        out[0] = '1'; out[1] = ':'; out[2] = '0'; out[3] = ','; out[4] = '0'; out[5] = 0;
        return out;
    }
    u128 p = vals[1];
    u128 g = vals[2];
    u128 bt = vals[3];
    u128 m1 = vals[4];
    u128 gamma1 = vals[5];
    u128 delta1 = vals[6];
    u128 m2 = vals[7];
    u128 gamma2 = vals[8];
    u128 delta2 = vals[9];
    if (!(gamma1 == gamma2)) {
        out[0] = '1'; out[1] = ':'; out[2] = '0'; out[3] = ','; out[4] = '0'; out[5] = 0;
        return out;
    }
    u128 v11 = mod_pow(bt, gamma1, p);
    u128 v12 = mod_pow(gamma1, delta1, p);
    u128 v1 = mod_mul(v11, v12, p);
    u128 r1 = mod_pow(g, m1, p);
    int ok1 = (v1 == r1);
    u128 v21 = mod_pow(bt, gamma2, p);
    u128 v22 = mod_pow(gamma2, delta2, p);
    u128 v2 = mod_mul(v21, v22, p);
    u128 r2 = mod_pow(g, m2, p);
    int ok2 = (v2 == r2);
    out[0] = '1'; out[1] = ':'; out[2] = ok1 ? '1' : '0'; out[3] = ','; out[4] = ok2 ? '1' : '0'; out[5] = 0;
    return out;
}

static const char* task2_recover(const char* block) {
    static char out[128];
    u128 vals[16];
    int n = parse_block_u128(block, 0, s_len_local(block), vals, 16);
    if (n < 10) {
        out[0] = '2'; out[1] = ':'; out[2] = '0'; out[3] = ';'; out[4] = '0'; out[5] = 0;
        return out;
    }
    u128 p = vals[1];
    u128 g = vals[2];
    u128 bt = vals[3];
    u128 m1 = vals[4];
    u128 gamma = vals[5];
    u128 d1 = vals[6];
    u128 m2 = vals[7];
    u128 gamma2 = vals[8];
    u128 d2 = vals[9];
    (void)g;
    (void)bt;
    (void)gamma2;
    u128 mod = p - 1;
    u128 num = mod_sub(m1, m2, mod);
    u128 den = mod_sub(d1, d2, mod);
    u128 gden = gcd_u128(den, mod);
    if (gden == 0 || (num % gden) != 0) {
        out[0] = '2'; out[1] = ':'; out[2] = '0'; out[3] = ';'; out[4] = '0'; out[5] = 0;
        return out;
    }
    u128 mod2 = mod / gden;
    u128 den2 = den / gden;
    u128 num2 = num / gden;
    u128 den2_inv = mod_inv_u128(den2, mod2);
    if (den2_inv == 0) {
        out[0] = '2'; out[1] = ':'; out[2] = '0'; out[3] = ';'; out[4] = '0'; out[5] = 0;
        return out;
    }
    u128 k0 = mod_mul(num2, den2_inv, mod2);
    u128 k = k0 % mod;
    u128 gammaM = gamma % mod;
    u128 inv_gamma = mod_inv_u128(gammaM, mod);
    if (inv_gamma == 0) {
        out[0] = '2'; out[1] = ':'; out[2] = '0'; out[3] = ';'; out[4] = '0'; out[5] = 0;
        return out;
    }
    u128 t = mod_mul(d1, k, mod);
    u128 num_a = mod_sub(m1, t, mod);
    u128 a = mod_mul(num_a, inv_gamma, mod);

    g_last_a = a;
    g_last_a_valid = 1;

    char abuf[64];
    char kbuf[64];
    u128_to_dec(a, abuf, 64);
    u128_to_dec(k, kbuf, 64);
    int pos = 0;
    out[pos++] = '2';
    out[pos++] = ':';
    int i = 0;
    while (abuf[i] && pos < (int)sizeof(out) - 1) out[pos++] = abuf[i++];
    out[pos++] = ';';
    i = 0;
    while (kbuf[i] && pos < (int)sizeof(out) - 1) out[pos++] = kbuf[i++];
    out[pos] = 0;
    return out;
}

static const char* task3_decrypt(const char* block, const char* alph) {
    static char out[256];
    u128 vals[16];
    int n = parse_block_u128(block, 0, s_len_local(block), vals, 16);
    if (n < 4) {
        out[0] = '3'; out[1] = ':'; out[2] = '0'; out[3] = 0;
        return out;
    }
    u128 p = vals[1];
    u128 a;
    if (g_last_a_valid) {
        a = g_last_a;
    } else {
        if (n < 3) {
            out[0] = '3'; out[1] = ':'; out[2] = '0'; out[3] = 0;
            return out;
        }
        a = vals[2];
    }
    u128 c1 = vals[n - 2];
    u128 c2 = vals[n - 1];
    u128 k = mod_pow(c1, a, p);
    u128 k_inv = mod_inv_u128(k, p);
    if (k_inv == 0) {
        out[0] = '3'; out[1] = ':'; out[2] = '0'; out[3] = 0;
        return out;
    }
    u128 m = mod_mul(c2, k_inv, p);
    char msg[192];
    num_to_text_utf8(alph, m, msg, sizeof msg);
    int pos = 0;
    out[pos++] = '3';
    out[pos++] = ':';
    int i = 0;
    while (msg[i] && pos < (int)sizeof(out) - 1) out[pos++] = msg[i++];
    out[pos] = 0;
    return out;
}

static const char* dispatch_task(int taskId, const char* block, const char* alph) {
    if (taskId == 1) return task1_verify(block);
    if (taskId == 2) return task2_recover(block);
    if (taskId == 3) return task3_decrypt(block, alph);
    static char empty[2] = {0,0};
    return empty;
}

const char* elGamalEntry(const char* alph, const char* encText, const char* frag) {
    static char out[768];
    if (!encText || !encText[0]) return "";
    if (!alph || !alph[0]) alph = "aąbcčdeęėfghiįyjklmnoprsštuųūvzž ";
    int idx[3][2];
    int blocks = parse_blocks(encText, idx);
    int tasks[3] = {0,0,0};
    int tcount = 0;
    g_last_a_valid = 0;
    g_last_a = 0;
    if (frag && frag[0]) {
        tcount = parse_frag_local(frag, tasks, 3);
    }
    if (tcount == 0) {
        tasks[0] = 1;
        tasks[1] = 2;
        tasks[2] = 3;
        tcount = 3;
    }
    int limit = blocks < tcount ? blocks : tcount;
    int pos = 0;
    for (int i = 0; i < limit; ++i) {
        int taskId = tasks[i];
        int s = idx[i][0];
        int e = idx[i][1];
        int len = e - s;
        if (len <= 0) continue;
        char buf[512];
        if (len >= (int)sizeof(buf)) len = (int)sizeof(buf) - 1;
        for (int k = 0; k < len; ++k) buf[k] = encText[s + k];
        buf[len] = 0;
        const char* r = dispatch_task(taskId, buf, alph);
        int j = 0;
        while (r[j] && pos < (int)sizeof(out) - 1) out[pos++] = r[j++];
        if (i + 1 < limit && pos < (int)sizeof(out) - 1) out[pos++] = '\n';
    }
    out[pos] = 0;
    return out;
}
