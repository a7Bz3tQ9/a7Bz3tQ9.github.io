#### 数位 DP

特点：范围特大，算的数与进制有关。

##### B1

CF908G New Year and Original Order（\*2800）

> 设 $S_n$ 表示将 $n$ 在十进制下所有数位从小到大排序后得到的数。如 $S(232)=223$，$S(50394)=3459$。
>
> 给出 $x$，求 $\sum_{i=1}^x S_i\pmod{10^9+7}$.
>
> $1\leq x\leq 10^{700}$。

数字 $d\in[1,9]$ 的贡献是 $c_d=\sum 10^i$ 的形式。朴素地求 $c_d$，DP 状态中不仅要记有多少 $=d$ 的数位，还要记有多少 $>d$ 的数位。复杂度爆炸。

$ans=\sum_{d=1}^9d\cdot c_d=\sum_{d=1}^9\sum_{i=d}^9 c_i$。

枚举 $d$ 考虑如何算 $\sum_{i=d}^9 c_i$，这实际上是所有 $\geq d$ 数位的 $10^i$ 之和。设 $f_{i,j,0/1}$ 表示考虑了前 $i$ 位，$\geq d$ 的数位有 $j$ 个，是否取到上界的方案数。将数位排序后，为 $\geq d$ 的是从最低位开始连续的一段，很好算。

##### B2

小练习

> 给出一个长度为 $n$ 的序列 $a$，$a_i\in [0,2^m)$。
>
> 求有多少个长度为 $n$ 的序列 $b$ 满足：$b\in[0,2^m)$，且 $(a_1\land b_1)\leq (a_2\land b_2)\leq\cdots\leq (a_n\land b_n)$，且 $(a_1\lor b_1)\geq (a_2\lor b_2)\geq \cdots\geq (a_n\lor b_n)$。
>
> $n,m\leq 30$。

注意到 $\land$ 只要考虑 $b_i$ 在 $a_i$ 中为 $1$ 的位是怎么取的，$\lor$ 只要考虑 $a_i$ 中为 $0$ 的位。不难发现，若 $a_i\land b_i$ 和 $a_i\lor b_i$ 确定，$b_i$ 就唯一确定了，所以两部分完全独立。

以 $\land$ 为例。由于 $\land$ 递增，那么肯定有某个分界点，最高位前半边是 $0$ 后半边是 $1$，然后若递归到前半边，又有次高位前半边是 $0$ 后半边是 $1$，后半边同理。于是区间 DP。

设 $f_{l,r,k}$ 表示当前考虑较低的 $k$ 位，要使得 $l\sim r$ 的 $\land$ 递增的方案数。转移考虑 $l\sim r$ 第 $k$ 位的 $\land$ 什么时候 $0\to 1$，$f_{l,r,k}\gets f_{l,p,k-1}\times f_{p+1,r,k-1}$。$ans_{\land}=f_{1,n,m}$。

类似的题目还有 ZR#370,P5985。

##### B3

ZR#615. 【19省选青岛集训】组合数

> 求：
> $$
> \sum_{i_1=l_1}^{r_1}\sum_{i_2=l_2}^{r_2}\cdots\sum_{i_n=l_n}^{r_n}\binom{m}{i_1+i_2+\cdots+i_n}\pmod p
> $$
> $1\leq n\leq 7$，$1\leq m\leq 10^{18}$，$1\leq l_i\leq r_i\leq 10^{17}$，$1\leq p\leq 10$ 且 $p$ 为质数。

注意到 $p$ 很小。转成 $p$ 进制然后数位 DP。

有上下界比较麻烦，可以容斥把下界去掉，$[l_i,r_i]=[0,r_i]-[0,l_i-1]$，复杂度乘个 $2^n$。现在假设下界全是 $0$，只有上界。容斥的 $\in[0,l_i-1],\in[0,r_i]$ 可以状压成二进制状态。

（技巧：容斥钦定不满足下界的集合 $S$，转化为只有上界。若 $i\in S$ 则 $x_i\in[0,l_i-1]$，否则 $x_i\in[0,r_i]$）

从低位到高位 DP，设 $f_{S,i,j}$ 表示容斥状态为 $S$，考虑到较低的 $i$ 位，$i\to i+1$ 进的位是 $j$。$i\to i+1$ 时，朴素的转移是 $\mathcal O(p^n)$ 枚举第 $i+1$ 位，算出新的 $S,j$。优化：轮廓线 DP 的思想。第 $i+1$ 位可以看成第 $i+1$ 行，每行有 $n$ 个数。这样不需要每次枚举出每行的状态。

```cpp
#include<bits/stdc++.h>
#define ll long long
using namespace std;
const int N=65;
int n,p,len,c[N][N],f[N][9][11][11][1<<7],vm[N],vi[N][N],ans;
ll m,l[N],r[N],lim[N];
int dfs(int x,int i,int vs,int o,int bad){ 	//考虑到第 x 个二进制位，第 i 个数，sum 的第 x 位为 vs，x->x+1 进的位是 o，只考虑 <=x 的二进制位时 > 上界的数的集合为 bad
	if(x>len) return !bad;
	if(i>n) return vs>vm[x]?0:1ll*c[vm[x]][vs]*dfs(x+1,1,o%p,o/p,bad)%p;
	int ans=f[x][i][vs][o][bad];
	if(~ans) return ans; ans=0;
	for(int v=0;v<p;v++){
		int tmp=bad;
		if(v>vi[x][i]) tmp|=1<<(i-1);
		else if(v<vi[x][i]) tmp&=((1<<n)-1)^(1<<(i-1));	//这位 < 上界，那即使更低位 > 上界，只考虑 <=x 的二进制位时第 i 个数也 < 上界
		ans=(ans+dfs(x,i+1,(vs+v)%p,o+(vs+v)/p,tmp))%p;
	}
	return f[x][i][vs][o][bad]=ans;
}
int calc(ll m){
	memset(f,-1,sizeof(f));
	for(int i=1;i<=len;i++){	//拆出 m 和 lim 在 p 进制下的每位
		vm[i]=m%p,m/=p;
		for(int j=1;j<=n;j++) vi[i][j]=lim[j]%p,lim[j]/=p; 
	}
	return dfs(1,1,0,0,0);
}
signed main(){
	scanf("%d%lld%d",&n,&m,&p),c[0][0]=1;
	for(int i=1;i<=n*p;i++){
		c[i][0]=1;
		for(int j=1;j<=i;j++) c[i][j]=(c[i-1][j]+c[i-1][j-1])%p;
	}
	for(ll i=1;i<=1e18;i*=p) len++;
	for(int i=1;i<=n;i++) scanf("%lld%lld",&l[i],&r[i]);
	for(int s=0;s<(1<<n);s++){	//容斥不满足下界的集合 s
		for(int i=1;i<=n;i++){
			if(s>>(i-1)&1) lim[i]=l[i]-1;	//容斥转化为只有上界 lim
			else lim[i]=r[i];
		}
		ans=(ans+1ll*(__builtin_popcount(s)&1?p-1:1)*calc(m)%p)%p;
	}
	printf("%d\n",ans);
	return 0;
}
```

##### P7961 [NOIP2021] 数列

> 给定整数 $n, m, k$，和一个长度为 $m + 1$ 的正整数数组 $v_0, v_1, \ldots, v_m$。
>
> 对于一个长度为 $n$，下标从 $1$ 开始且每个元素均不超过 $m$ 的非负整数序列 $\{a_i\}$，我们定义它的权值为 $v_{a_1} \times v_{a_2} \times \cdots \times v_{a_n}$。
>
> 当这样的序列 $\{a_i\}$ 满足整数 $S = 2^{a_1} + 2^{a_2} + \cdots + 2^{a_n}$ 的二进制表示中 $1$ 的个数不超过 $k$ 时，我们认为 $\{a_i\}$ 是一个合法序列。
>
> 计算所有合法序列 $\{a_i\}$ 的权值和对 $998244353$ 取模的结果。



##### CF582D Number of Binominal Coefficients

> 给定质数 $p$ 和整数 $\alpha,A$，求满足 $0\leq k\leq n\leq A$ 且 $p^{\alpha}\mid \dbinom n k$ 的数对 $(n,k)$ 的个数。
>
> $p,\alpha\leq 10^9$，$A\leq 10^{1000}$，答案对 $10^9+7$ 取模。

Kummer 定理：$\large\binom{n+m}{m}$ 中质因子 $p$ 的幂次为 $n,m$ 在 $p$ 进制下相加的进位次数。

> 证明：$x!$ 中 $p$ 的次数为 $\sum_{i\geq 1}\lfloor\frac{x}{p^i}\rfloor$。
>
> 考虑 $\lfloor\frac{x}{p^i}\rfloor$ 就是 $x$ 在 $p$ 进制下砍掉后面的 $0\sim i-1$ 位。考虑第 $i$ 位的贡献 $\lfloor\frac{n+m}{p^i}\rfloor-\lfloor\frac{n}{p^i}\rfloor-\lfloor\frac{m}{p^i}\rfloor$，思考“加起来再砍”和“砍掉再加”，可得贡献为 $1$ 当且仅当 $n+m$ 过程中第 $i-1$ 位进位到第 $i$ 位，否则贡献为 $0$。

首先在做这道题之前你需要知道库默尔（Kummer）定理：对于质数 $p$ 及 $n,k$，最大的满足 $p^{\alpha}\mid \binom n k$ 的 $\alpha$ 为 $k$ 与 $n-k$ 在 $p$ 进制下相加的进位次数。或者说，质数 $p$ 进制下 $x+y$ 的进位次数等于 $\binom{x+y}{x}$ 的标准分解式中 $p$ 的次数。证明考虑扩展 Lucas 定理，此处略。

问题转化为求 $0\leq x+y\leq A$ 且 $x+y$ 在 $p$ 进制下的进位次数 $\geq\alpha$ 的非负整数对 $(x,y)$ 的个数。

然后就可以数位 DP 了。题目中 $\alpha\leq 10^9$ 是假的，如果 $\alpha>\log_p A$ 那答案显然为 $0$（因为进位次数肯定 $\leq \log_pA$）。首先将 $A$ 用 $p$ 进制表示，我们设 $dp_{i,j,0/1,0/1}$ 表示考虑了最高的 $i$ 位，当前进位了 $j$ 次，上一位（第 $i+1$ 高的位）是否产生进位，当前是否达到上界，考虑转移，假设 $A$  的第 $i+1$ 位的值为 $c$，我们要决策 $k$ 的第 $i+1$ 位的值 $a$ 与 $n-k$ 第 $i+1$ 位的值 $b$，那么有转移：

- $dp_{i+1,j,0,0}$：
  - 如果从 $dp_{i,j,0,0}$ 转移来那么需满足 $a+b<p$，方案数 $\frac{p(p+1)}{2}$（考虑隔板法）。
  - 如果从 $dp_{i,j,0,1}$ 转移来那么需满足 $a+b<c$，方案数 $\frac{c(c+1)}{2}$。
  - 如果从 $dp_{i,j,1,0}$ 转移来那么需满足 $a+b\geq p$（第 $i+1$ 位产生进位），方案数 $\frac{p(p-1)}{2}$（$p-a+p-b\leq p$，$p-a,p-b\geq 1$）。
  - 如果从 $dp_{i,j,1,1}$ 转移来那么需满足 $p\leq a+b<p+c$，方案数 $\frac{(p+c)(p+c+1)}{2}-\frac{p(p+1)}{2}=\frac{c(2n-c-1)}{2}$。
- $dp_{i+1,j,0,1}$：
  - 如果从 $dp_{i,j,0,1}$ 转移来那么需满足 $a+b=c$，方案数 $c+1$。
  - 如果从 $dp_{i,j,1,1}$ 转移来那么需满足 $a+b=p+c$，方案数 $p-c-1$（$0\leq a,p+c-a<p$，$c<a<p$）。
- $dp_{i+1,j,1,0}$（第 $i+2$ 位进到第 $i+1$ 位）：
  - 如果从 $dp_{i,j,0,0}$ 转移来那么需满足 $a+b<p-1$，方案数 $\frac{p(p-1)}{2}$。
  - 如果从 $dp_{i,j,0,1}$ 转移来那么需满足 $a+b<c-1$，方案数 $\frac{c(c-1)}{2}$。
  - 如果从 $dp_{i,j,1,0}$ 转移来那么需满足 $a+b\geq p-1$，方案数 $\frac{p(p+1)}{2}$。
  - 如果从 $dp_{i,j,1,1}$ 转移来那么需满足 $p\leq a+b+1<p+c$，方案数 $\frac{(p+c)(p+c-1)}{2}-\frac{p(p-1)}{2}=\frac{c(2n-c+1)}{2}$。
- $dp_{i+1,j,1,1}$：
  - 如果从 $dp_{i,j,0,1}$ 转移来那么需满足 $a+b+1=c$，方案数 $c$。
  - 如果从 $dp_{i,j,1,1}$ 转移来那么需满足 $a+b+1=p+c$，方案数 $p-c$。

算下贡献转移一下即可。时间复杂度 $\mathcal O(\log_k^2 A)$。

最好使用滚动数组优化。

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N=4e3+5,mod=1e9+7;
int p,a,m,A[N],len,x[N],lst,cur,f[2][N][2][2],ans;
char s[N];
signed main(){
	scanf("%d%d%s",&p,&a,s+1),len=strlen(s+1);
	if(a>4e3) puts("0"),exit(0);
	for(int i=1;i<=len;i++) A[len-i+1]=s[i]-'0';
	while(len){
		long long cur=0;
		for(int i=len;i>=1;i--) cur=cur*10+A[i],A[i]=cur/p,cur%=p;
		x[++m]=cur;
		if(!A[len]) len--;
	}
	f[cur=1][0][0][1]=1;
	for(int i=m;i>=1;i--){
		swap(cur,lst),memset(f[cur],0,sizeof(f[cur]));
		int c1=1ll*p*(p+1)/2%mod,c2=1ll*(x[i]+1)*x[i]/2%mod;
		int c3=1ll*(p-1)*p/2%mod,c4=1ll*x[i]*(p*2-x[i]-1)/2%mod;
		int c5=1ll*(x[i]-1)*x[i]/2%mod,c6=1ll*x[i]*(p*2-x[i]+1)/2%mod;
		for(int j=0;j<=m-i+1;j++){
			int f0=f[lst][j][0][0],f1=f[lst][j][0][1];
			int f2=f[lst][j][1][0],f3=f[lst][j][1][1];
			f[cur][j][0][0]=(1ll*f0*c1%mod+1ll*f1*c2%mod+1ll*f2*c3%mod+1ll*f3*c4%mod)%mod;
			f[cur][j][0][1]=(1ll*(x[i]+1)*f1%mod+1ll*(p-x[i]-1)*f3%mod)%mod;
			f[cur][j+1][1][0]=(1ll*f0*c3%mod+1ll*f1*c5%mod+1ll*f2*c1%mod+1ll*f3*c6%mod)%mod;
			f[cur][j+1][1][1]=(1ll*f1*x[i]%mod+1ll*f3*(p-x[i])%mod)%mod;
		}
	}
	for(int i=a;i<=m;i++)
		ans=((ans+f[cur][i][0][0])%mod+f[cur][i][0][1])%mod;
	printf("%d\n",ans);
	return 0;
}
```

