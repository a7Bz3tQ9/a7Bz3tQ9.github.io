[1](https://www.cnblogs.com/maoyiting/p/13356633.html)

<h3>一、关于数位 dp</h3>
<p>有时候我们会遇到某类问题，它所统计的对象具有某些性质，答案在限制/贡献上与统计对象的数位之间有着密切的关系，有可能是数位之间联系的形式，也有可能是数位之间相互独立的形式。（如求满足条件的第 K 小的数是多少，或者求在区间 [L,R] 内有多少个满足限制条件的数等）</p>
<p>常见的在 dp 状态中需要记的信息：当前位数、与上界之间的关系（从高到低做这个信息为 0/1，即当前与上界相等/小于上界。往往数位 dp 的对象是 0 到某个上界 R，为了统计这个范围的信息，我们需要保证从高位往低位做的过程中，这个数始终是小于等于这个上界的。从低到高做这个信息为 0/1/2），是否处于前导零状态等，更多的是跟题目条件有关的信息（包括题目的限制/贡献）。</p>
<p><strong>写法:</strong></p>
<ul>
<li><span>每一维的信息用循环遍历到，转移时判每一种合法/不合法的情况。</span>(缺点: 容易漏情况)</li>
<li><span>手动转移。手展合法的情况。(缺点: 难写难查)</span></li>
</ul>
<h3>二、例题</h3>
<h4>1. HDU3652 B-Number</h4>
<p><strong>题目大意：</strong>问 1~N 中所有含有 13 并且能被 13 整除的数的个数。N&le;10<sup>9</sup>。</p>
<div><strong>Solution：</strong>&ldquo;含有 13&rdquo;，即符合条件的数有相邻的两个数位：1、3；&ldquo;能被 13 整除&rdquo;这个信息可以通过从高到低按数位计算。因此，在这道题中的两个限制，都和数位之间有着密切的关系。</div>
<p>设 dp[i][j][k][t=0/1] 表示当前第 i 位，上一位（即第 i+1 位）是 j，对 13 取模余数为 k，小于/等于上界（t 表示与上界之间的关系）的方案数。（针对前缀的个数，对 13 取模也是对前缀 13 取模）</p>
<p>我们需要考虑 i-1 位的数位，这个数位枚举的范围由 t 决定。若 t=0，即此时它小于上界，则当前这个数位枚举的范围为 0~9；若 t=1，即此时它等于上界，则当前这个数位枚举的范围为 0~N 的当前数位。设枚举到的这个数位为 c，按照定义，则可以从 dp[i-1][c][(k*10+c)%13][t=1&amp;&amp;c=N 的当前位] 转移到 dp[i][j][k][t]。</p>
<p>由于统计的是数的个数，因此转移为： dp[i][j][k][t]+=dp[i-1][c][(k*10+c)%13][t=1&amp;&amp;c=N 的当前位]。</p>
<p>以上的 dp 状态对于这道题还有遗漏的地方，还需再开一维状态记录目前是否出现了 13。此时还可以缩减状态：用 0 表示之前什么都没有出现过，1 表示上一位以 1 结尾，2 表示已出现了 13。</p>
<p><strong>Code：</strong></p>
<p>数位 dp 的第一种写法：</p>
<div class="cnblogs_code">
<pre><span style="color: #0000ff;">void</span><span style="color: #000000;"> solve(){
    memset(dp,</span><span style="color: #800080;">0</span>,<span style="color: #0000ff;">sizeof</span><span style="color: #000000;">(dp));
    dp[</span><span style="color: #800080;">0</span>][<span style="color: #800080;">0</span>][<span style="color: #800080;">1</span>][<span style="color: #800080;">0</span>]=<span style="color: #800080;">1</span><span style="color: #000000;">;
    </span><span style="color: #0000ff;">for</span>(<span style="color: #0000ff;">int</span> i=<span style="color: #800080;">0</span>;i&lt;len;i++)    <span style="color: #008000;">//</span><span style="color: #008000;">数位 </span>
        <span style="color: #0000ff;">for</span>(<span style="color: #0000ff;">int</span> j=<span style="color: #800080;">0</span>;j&lt;=<span style="color: #800080;">12</span>;j++)    <span style="color: #008000;">//</span><span style="color: #008000;">mod 13 的值 </span>
            <span style="color: #0000ff;">for</span>(<span style="color: #0000ff;">int</span> k=<span style="color: #800080;">0</span>;k&lt;=<span style="color: #800080;">1</span>;k++)    <span style="color: #008000;">//</span><span style="color: #008000;">与上界之间的关系 </span>
                <span style="color: #0000ff;">for</span>(<span style="color: #0000ff;">int</span> m=<span style="color: #800080;">0</span>;m&lt;=<span style="color: #800080;">2</span>;m++)    <span style="color: #008000;">//</span><span style="color: #008000;">处理是否出现了 13。0 表示之前什么都没有出现过，1 表示上一位以 1 结尾，2 表示已出现了 13。</span>
                    <span style="color: #0000ff;">if</span>(dp[i][j][k][m]!=<span style="color: #800080;">0</span><span style="color: #000000;">){
                        </span><span style="color: #0000ff;">int</span> end=(k==<span style="color: #800080;">1</span>)?s[i]-<span style="color: #800000;">'</span><span style="color: #800000;">0</span><span style="color: #800000;">'</span>:<span style="color: #800080;">9</span>;    <span style="color: #008000;">//</span><span style="color: #008000;">数位上界 </span>
                        <span style="color: #0000ff;">for</span>(<span style="color: #0000ff;">int</span> x=<span style="color: #800080;">0</span>;x&lt;=end;x++)    <span style="color: #008000;">//</span><span style="color: #008000;">当前数位 </span>
                            dp[i+<span style="color: #800080;">1</span>][(j*<span style="color: #800080;">10</span>+x)%<span style="color: #800080;">13</span>][k==<span style="color: #800080;">1</span>&amp;x==r][(x==<span style="color: #800080;">1</span>&amp;&amp;m!=<span style="color: #800080;">2</span>)?<span style="color: #800080;">1</span>:((m==<span style="color: #800080;">2</span>||(x==<span style="color: #800080;">3</span>&amp;&amp;m==<span style="color: #800080;">1</span>))?<span style="color: #800080;">2</span>:<span style="color: #800080;">0</span>)]+=<span style="color: #000000;">dp[i][j][k][m];
                    }
} </span><span style="color: #008000;">//</span><span style="color: #008000;">ans=sigma dp[len][0][0/1][2] </span></pre>
</div>
<p>&nbsp;数位 dp 的另一种写法：（记忆化搜索）</p>
<div class="cnblogs_code">
<pre>#include&lt;bits/stdc++.h&gt;
<span style="color: #0000ff;">#define</span> int long long
<span style="color: #0000ff;">using</span> <span style="color: #0000ff;">namespace</span><span style="color: #000000;"> std;
</span><span style="color: #0000ff;">const</span> <span style="color: #0000ff;">int</span> N=<span style="color: #800080;">15</span><span style="color: #000000;">;
</span><span style="color: #0000ff;">int</span> n,f[N][N][N][<span style="color: #800080;">2</span>][<span style="color: #800080;">2</span>],a[<span style="color: #800080;">20</span><span style="color: #000000;">];
</span><span style="color: #0000ff;">int</span> dfs(<span style="color: #0000ff;">int</span> i,<span style="color: #0000ff;">int</span> last,<span style="color: #0000ff;">int</span> p,<span style="color: #0000ff;">bool</span> have13,<span style="color: #0000ff;">bool</span> less){    <span style="color: #008000;">//</span><span style="color: #008000;">i:数位  last:上一个数位  p:mod 13的值  have13:是否含有13  less:与上界之间的关系 </span>
    <span style="color: #0000ff;">if</span>(!i) <span style="color: #0000ff;">return</span> have13&amp;&amp;(p==<span style="color: #800080;">0</span><span style="color: #000000;">);
    </span><span style="color: #0000ff;">if</span>(!less&amp;&amp;f[i][last][p][have13][less]!=-<span style="color: #800080;">1</span>) <span style="color: #0000ff;">return</span><span style="color: #000000;"> f[i][last][p][have13][less];
    </span><span style="color: #0000ff;">int</span> end=less?a[i]:<span style="color: #800080;">9</span>,ans=<span style="color: #800080;">0</span><span style="color: #000000;">;
    </span><span style="color: #0000ff;">for</span>(<span style="color: #0000ff;">int</span> j=<span style="color: #800080;">0</span>;j&lt;=end;j++)    <span style="color: #008000;">//</span><span style="color: #008000;">当前数位 </span>
        ans+=dfs(i-<span style="color: #800080;">1</span>,j,(p*<span style="color: #800080;">10</span>+j)%<span style="color: #800080;">13</span>,have13||(last==<span style="color: #800080;">1</span>&amp;&amp;j==<span style="color: #800080;">3</span>),less&amp;&amp;j==<span style="color: #000000;">end);
    </span><span style="color: #0000ff;">if</span>(!less) f[i][last][p][have13][less]=<span style="color: #000000;">ans;
    </span><span style="color: #0000ff;">return</span><span style="color: #000000;"> ans;
}
</span><span style="color: #0000ff;">int</span> calc(<span style="color: #0000ff;">int</span><span style="color: #000000;"> x){
    </span><span style="color: #0000ff;">int</span> n=<span style="color: #800080;">0</span><span style="color: #000000;">;
    </span><span style="color: #0000ff;">while</span>(x) a[++n]=x%<span style="color: #800080;">10</span>,x/=<span style="color: #800080;">10</span><span style="color: #000000;">;
    </span><span style="color: #0000ff;">return</span> dfs(n,<span style="color: #800080;">0</span>,<span style="color: #800080;">0</span>,<span style="color: #800080;">0</span>,<span style="color: #800080;">1</span><span style="color: #000000;">);
}
signed main(){
    </span><span style="color: #0000ff;">while</span>(~scanf(<span style="color: #800000;">"</span><span style="color: #800000;">%lld</span><span style="color: #800000;">"</span>,&amp;<span style="color: #000000;">n)){
        memset(f,</span>-<span style="color: #800080;">1</span>,<span style="color: #0000ff;">sizeof</span><span style="color: #000000;">(f));
        printf(</span><span style="color: #800000;">"</span><span style="color: #800000;">%lld\n</span><span style="color: #800000;">"</span><span style="color: #000000;">,calc(n));
    }
    </span><span style="color: #0000ff;">return</span> <span style="color: #800080;">0</span><span style="color: #000000;">;
} </span></pre>
</div>
<h4>2.&nbsp;Luogu P4317 花神的数论题</h4>
<p><strong>题目大意：</strong>问 1~N 中所有数转化为二进制后数位中 1 的个数之积。N&le;10<sup>15</sup>。</p>
<p><strong>Solution：</strong></p>
<p>令 dp[i][j][k] 表示 dp 到第 i 位，数位中有 j 个 1，跟上界 N 之间的大小关系为 k（0相等，1小于）的方案数。</p>
<p>第 i 位取 0：dp[i-1][j][k]&rarr;dp[i][j][k|N(i)]</p>
<p>（若 N(i) 为 0，则与上界的大小关系不变；若 N(i) 为1，而第 i 位取了 0，则已经小于上界）</p>
<p>第 i 位取 1：dp[i-1][j][k]*max(k,N(i))&rarr;dp[i][j+1][k]</p>
<p>（在这种情况下，若 k=N(i)=0是不合法的，也就是说之前已经达到上界了，并且在这一位是0，但第 i 位取了1，就超过了上界，所以要乘以 max(k,N(i))）</p>
<p>其中 N(i) 表示上界 N 在二进制下第 i 位的取值。</p>
<p>最终的答案为：对于任意 1&le;j&le;n，<span id="MathJax-Span-300" class="mrow"><span id="MathJax-Span-301" class="mi">j<sup>dp[n][j][0]+dp[n][j][1]</sup> 的乘积。</span></span></p>
<p><strong>Code：</strong></p>
<p>写法1：</p>
<div class="cnblogs_code">
<pre>#include&lt;bits/stdc++.h&gt;
<span style="color: #0000ff;">#define</span> int long long
<span style="color: #0000ff;">using</span> <span style="color: #0000ff;">namespace</span><span style="color: #000000;"> std;
</span><span style="color: #0000ff;">const</span> <span style="color: #0000ff;">int</span> N=<span style="color: #800080;">60</span>,mod=1e7+<span style="color: #800080;">7</span><span style="color: #000000;">;
</span><span style="color: #0000ff;">int</span> n,f[N][N][<span style="color: #800080;">2</span>],ans=<span style="color: #800080;">1</span><span style="color: #000000;">,cnt,a[N];
</span><span style="color: #0000ff;">int</span> mul(<span style="color: #0000ff;">int</span> x,<span style="color: #0000ff;">int</span> a,<span style="color: #0000ff;">int</span> mod){    <span style="color: #008000;">//</span><span style="color: #008000;">快速幂 </span>
    <span style="color: #0000ff;">int</span> ans=mod!=<span style="color: #800080;">1</span><span style="color: #000000;">;
    </span><span style="color: #0000ff;">for</span>(x%=mod;a;a&gt;&gt;=<span style="color: #800080;">1</span>,x=x*x%<span style="color: #000000;">mod)
        </span><span style="color: #0000ff;">if</span>(a&amp;<span style="color: #800080;">1</span>) ans=ans*x%<span style="color: #000000;">mod;
    </span><span style="color: #0000ff;">return</span><span style="color: #000000;"> ans;
}
signed main(){
    scanf(</span><span style="color: #800000;">"</span><span style="color: #800000;">%lld</span><span style="color: #800000;">"</span>,&amp;<span style="color: #000000;">n);
    </span><span style="color: #0000ff;">while</span>(n) a[++cnt]=n%<span style="color: #800080;">2</span>,n/=<span style="color: #800080;">2</span><span style="color: #000000;">;
    reverse(a</span>+<span style="color: #800080;">1</span>,a+<span style="color: #800080;">1</span>+cnt),f[<span style="color: #800080;">0</span>][<span style="color: #800080;">0</span>][<span style="color: #800080;">0</span>]=<span style="color: #800080;">1</span><span style="color: #000000;">;
    </span><span style="color: #0000ff;">for</span>(<span style="color: #0000ff;">int</span> i=<span style="color: #800080;">1</span>;i&lt;=cnt;i++<span style="color: #000000;">)
        </span><span style="color: #0000ff;">for</span>(<span style="color: #0000ff;">int</span> j=<span style="color: #800080;">0</span>;j&lt;i;j++<span style="color: #000000;">)
            </span><span style="color: #0000ff;">for</span>(<span style="color: #0000ff;">int</span> k=<span style="color: #800080;">0</span>;k&lt;=<span style="color: #800080;">1</span>;k++<span style="color: #000000;">){
                f[i][j][k</span>|a[i]]+=f[i-<span style="color: #800080;">1</span><span style="color: #000000;">][j][k];
                f[i][j</span>+<span style="color: #800080;">1</span>][k]+=f[i-<span style="color: #800080;">1</span>][j][k]*<span style="color: #000000;">max(k,a[i]);
            }
    </span><span style="color: #0000ff;">for</span>(<span style="color: #0000ff;">int</span> j=<span style="color: #800080;">1</span>;j&lt;=cnt;j++<span style="color: #000000;">)
        ans</span>=ans*mul(j,f[cnt][j][<span style="color: #800080;">0</span>]+f[cnt][j][<span style="color: #800080;">1</span>],mod)%<span style="color: #000000;">mod;
    printf(</span><span style="color: #800000;">"</span><span style="color: #800000;">%lld\n</span><span style="color: #800000;">"</span><span style="color: #000000;">,ans);
    </span><span style="color: #0000ff;">return</span> <span style="color: #800080;">0</span><span style="color: #000000;">;
}</span></pre>
</div>
<p>写法2：</p>
<div class="cnblogs_code">
<pre>#include&lt;bits/stdc++.h&gt;
<span style="color: #0000ff;">#define</span> int long long
<span style="color: #0000ff;">using</span> <span style="color: #0000ff;">namespace</span><span style="color: #000000;"> std; 
</span><span style="color: #0000ff;">const</span> <span style="color: #0000ff;">int</span> N=<span style="color: #800080;">60</span>,mod=1e7+<span style="color: #800080;">7</span><span style="color: #000000;">;
</span><span style="color: #0000ff;">int</span> n,f[N][N][<span style="color: #800080;">2</span><span style="color: #000000;">],a[N],t;
</span><span style="color: #0000ff;">int</span> mul(<span style="color: #0000ff;">int</span> x,<span style="color: #0000ff;">int</span> n,<span style="color: #0000ff;">int</span> mod){    <span style="color: #008000;">//</span><span style="color: #008000;">快速幂 </span>
    <span style="color: #0000ff;">int</span> ans=mod!=<span style="color: #800080;">1</span><span style="color: #000000;">;
    </span><span style="color: #0000ff;">for</span>(x%=mod;n;n&gt;&gt;=<span style="color: #800080;">1</span>,x=x*x%<span style="color: #000000;">mod)
        </span><span style="color: #0000ff;">if</span>(n&amp;<span style="color: #800080;">1</span>) ans=ans*x%<span style="color: #000000;">mod;
    </span><span style="color: #0000ff;">return</span><span style="color: #000000;"> ans;
}
</span><span style="color: #0000ff;">int</span> dfs(<span style="color: #0000ff;">int</span> i,<span style="color: #0000ff;">int</span> cnt,<span style="color: #0000ff;">bool</span> less){    <span style="color: #008000;">//</span><span style="color: #008000;">i:数位  cnt:1的个数  less:与上界之间的关系 </span>
    <span style="color: #0000ff;">if</span>(!i) <span style="color: #0000ff;">return</span> t==<span style="color: #000000;">cnt;
    </span><span style="color: #0000ff;">if</span>(!less&amp;&amp;f[i][cnt][less]!=-<span style="color: #800080;">1</span>) <span style="color: #0000ff;">return</span><span style="color: #000000;"> f[i][cnt][less];
    </span><span style="color: #0000ff;">int</span> end=less?a[i]:<span style="color: #800080;">1</span>,ans=<span style="color: #800080;">0</span><span style="color: #000000;">;
    </span><span style="color: #0000ff;">for</span>(<span style="color: #0000ff;">int</span> j=<span style="color: #800080;">0</span>;j&lt;=end;j++)    <span style="color: #008000;">//</span><span style="color: #008000;">当前数位 </span>
        ans+=dfs(i-<span style="color: #800080;">1</span>,cnt+(j==<span style="color: #800080;">1</span>),less&amp;&amp;j==<span style="color: #000000;">end);
    </span><span style="color: #0000ff;">if</span>(!less) f[i][cnt][less]=<span style="color: #000000;">ans;
    </span><span style="color: #0000ff;">return</span><span style="color: #000000;"> ans;
}
</span><span style="color: #0000ff;">int</span> calc(<span style="color: #0000ff;">int</span><span style="color: #000000;"> x){
    </span><span style="color: #0000ff;">int</span> n=<span style="color: #800080;">0</span><span style="color: #000000;">;
    </span><span style="color: #0000ff;">while</span>(x) a[++n]=x%<span style="color: #800080;">2</span>,x/=<span style="color: #800080;">2</span><span style="color: #000000;">;
    </span><span style="color: #0000ff;">int</span> ans=<span style="color: #800080;">1</span><span style="color: #000000;">;
    </span><span style="color: #0000ff;">for</span>(<span style="color: #0000ff;">int</span> i=<span style="color: #800080;">1</span>;i&lt;=n;i++){    <span style="color: #008000;">//</span><span style="color: #008000;">枚举1的个数 </span>
        memset(f,-<span style="color: #800080;">1</span>,<span style="color: #0000ff;">sizeof</span><span style="color: #000000;">(f));
        t</span>=i,ans=ans*mul(i,dfs(n,<span style="color: #800080;">0</span>,<span style="color: #800080;">1</span>),mod)%<span style="color: #000000;">mod;
    }
    </span><span style="color: #0000ff;">return</span><span style="color: #000000;"> ans;
}
signed main(){
    scanf(</span><span style="color: #800000;">"</span><span style="color: #800000;">%lld</span><span style="color: #800000;">"</span>,&amp;<span style="color: #000000;">n);
    printf(</span><span style="color: #800000;">"</span><span style="color: #800000;">%lld\n</span><span style="color: #800000;">"</span><span style="color: #000000;">,calc(n));
    </span><span style="color: #0000ff;">return</span> <span style="color: #800080;">0</span><span style="color: #000000;">;
}</span></pre>
</div>
<h4>3. Luogu P2602 数字计数</h4>
<p><strong>题目大意：</strong>给定两个正整数 a 和 b，求在 [a,b] 中的所有整数中，每个数码（digit）各出现了多少次。a&le;b&le;10<sup>12</sup>。</p>
<p><strong>Solution：</strong></p>
<p>做法一：预处理出所有 0~10<sup>x</sup>-1 的答案，将询问中给出的 [a, b] 转化为 [0, b] 的答案减去 [0,a-1] 的答案，对于最高位上界为 R 的任务，可以利用预处理出的结果统计最高位在 0~R-1 范围内的答案；将最高位独自的贡献统计了之后就可只考虑其他数位，问题转化到了一个结构相同但数位减少了一的任务上去。</p>
<p>做法二：</p>
<p>令 f[i][0/1] 表示已经做到了第 i 位，与上界之间的大小关系（0相等，1小于）时的答案总和。由于在后续的统计当中单个数位贡献的权还跟满足该种状态的数字个数有关，因此我们需要额外记录一下数字个数 g[i][0/1]。(在本题中这个值可以直接计算)</p>
<h4>4. HDU4734 f(x)</h4>
<p><strong>题目大意：</strong>定义一个十进制数 N 的权值 F(N) 为其各个数位乘上 2 的（后面位数）次幂之和。给出 A,B。问 0~B 中权值不超过 F(A) 的数个数。A,B&le;10<sup>9</sup>。</p>
<p><strong>Solution：</strong></p>
<p>Sum 的范围大约在 5k 的样子，直接存下来即可。</p>
<p>令 dp[i][j][0/1] 表示 dp 到数字的第 i 位，每个数位对 F 值贡献之和为 j 的方案数。</p>
<div class="cnblogs_code">
<pre>#include&lt;bits/stdc++.h&gt;
<span style="color: #0000ff;">#define</span> int long long
<span style="color: #0000ff;">using</span> <span style="color: #0000ff;">namespace</span><span style="color: #000000;"> std;
</span><span style="color: #0000ff;">const</span> <span style="color: #0000ff;">int</span> N=<span style="color: #800080;">15</span>,M=2e4+<span style="color: #800080;">5</span><span style="color: #000000;">;
</span><span style="color: #0000ff;">int</span> t,A,B,f[N][M][<span style="color: #800080;">2</span><span style="color: #000000;">],a[N]; 
</span><span style="color: #0000ff;">int</span> F(<span style="color: #0000ff;">int</span> x){    <span style="color: #008000;">//</span><span style="color: #008000;">计算 F(x) </span>
    <span style="color: #0000ff;">int</span> cnt=<span style="color: #800080;">0</span>,y=<span style="color: #800080;">1</span><span style="color: #000000;">; 
    </span><span style="color: #0000ff;">while</span>(x) cnt+=x%<span style="color: #800080;">10</span>*y,x/=<span style="color: #800080;">10</span>,y&lt;&lt;=<span style="color: #800080;">1</span><span style="color: #000000;">;
    </span><span style="color: #0000ff;">return</span><span style="color: #000000;"> cnt;
}
</span><span style="color: #0000ff;">int</span> dfs(<span style="color: #0000ff;">int</span> i,<span style="color: #0000ff;">int</span> sum,<span style="color: #0000ff;">bool</span> less){    <span style="color: #008000;">//</span><span style="color: #008000;">i:数位  j:每个数位对 F 值贡献之和  less:与上界之间的关系 </span>
    <span style="color: #0000ff;">if</span>(!i) <span style="color: #0000ff;">return</span> sum&gt;=<span style="color: #800080;">0</span><span style="color: #000000;">;
    </span><span style="color: #0000ff;">if</span>(sum&lt;<span style="color: #800080;">0</span>) <span style="color: #0000ff;">return</span> <span style="color: #800080;">0</span><span style="color: #000000;">;
    </span><span style="color: #0000ff;">if</span>(!less&amp;&amp;f[i][sum][less]!=-<span style="color: #800080;">1</span>) <span style="color: #0000ff;">return</span><span style="color: #000000;"> f[i][sum][less]; 
    </span><span style="color: #0000ff;">int</span> end=less?a[i]:<span style="color: #800080;">9</span>,ans=<span style="color: #800080;">0</span><span style="color: #000000;">;
    </span><span style="color: #0000ff;">for</span>(<span style="color: #0000ff;">int</span> j=<span style="color: #800080;">0</span>;j&lt;=end;j++)    <span style="color: #008000;">//</span><span style="color: #008000;">当前数位 </span>
        ans+=dfs(i-<span style="color: #800080;">1</span>,sum-j*(<span style="color: #800080;">1</span>&lt;&lt;(i-<span style="color: #800080;">1</span>)),less&amp;&amp;j==<span style="color: #000000;">end);
    </span><span style="color: #0000ff;">if</span>(!less) f[i][sum][less]=<span style="color: #000000;">ans;
    </span><span style="color: #0000ff;">return</span><span style="color: #000000;"> ans;
}
</span><span style="color: #0000ff;">int</span> calc(<span style="color: #0000ff;">int</span><span style="color: #000000;"> x){
    </span><span style="color: #0000ff;">int</span> n=<span style="color: #800080;">0</span><span style="color: #000000;">;
    </span><span style="color: #0000ff;">while</span>(x) a[++n]=x%<span style="color: #800080;">10</span>,x/=<span style="color: #800080;">10</span><span style="color: #000000;">;
    </span><span style="color: #0000ff;">return</span> dfs(n,F(A),<span style="color: #800080;">1</span><span style="color: #000000;">);
}
signed main(){
    memset(f,</span>-<span style="color: #800080;">1</span>,<span style="color: #0000ff;">sizeof</span><span style="color: #000000;">(f));
    scanf(</span><span style="color: #800000;">"</span><span style="color: #800000;">%lld</span><span style="color: #800000;">"</span>,&amp;<span style="color: #000000;">t);
    </span><span style="color: #0000ff;">for</span>(<span style="color: #0000ff;">int</span> k=<span style="color: #800080;">1</span>;k&lt;=t;k++<span style="color: #000000;">){
        scanf(</span><span style="color: #800000;">"</span><span style="color: #800000;">%lld%lld</span><span style="color: #800000;">"</span>,&amp;A,&amp;<span style="color: #000000;">B);
        printf(</span><span style="color: #800000;">"</span><span style="color: #800000;">Case #%lld: %lld\n</span><span style="color: #800000;">"</span><span style="color: #000000;">,k,calc(B));
    }
    </span><span style="color: #0000ff;">return</span> <span style="color: #800080;">0</span><span style="color: #000000;">;
} </span></pre>
</div>
<h4>5. Codeforces 55D Beautiful Number</h4>
<p><strong>题目大意</strong>：给定两个正整数 a 和 b，求在 [a,b] 中的所有整数中，有多少个数能够整除它自身的所有非零数位。a&le;b&le;10<sup>18</sup>。</p>
<p><strong>Solution：</strong></p>
<p>一种直接的想法是维护当前数模 1,2,...,9 的余数。每次乘十再加上新的数位后取个模就是新的状态。另外再维护一个状态表示 1,2,..,9 这些数位有哪些出现过。状态数18*2*9!*2<sup>9</sup>&asymp;6<sup>9</sup>。</p>
<p>注意到若是我们维护了模 9 的余数，便自然能够推出模 3 的余数。类似这样的冗余状态有不少。</p>
<p>按照上面的思路，一个数能被其所有的非零数位整除，即它能被所有的非零数位的最小公倍数整除，这个最小公倍数的最大值显然是 1 到 9 的最小公倍数 2520 ，然后就可以对 2520 的模进行状态转移。 我们可以维护这个数本身模 lcm(1,2,...,9)=2520 的余数。从这个值可推出模所有 1,2,..,9 数位的余数。这一步优化将状态从 9!&rarr;2520。目前的状态数为 18*2*2520*512&asymp;4<sup>7</sup>。</p>
<p>dp[i][p][j][k] 表示当前 dp 到第 i 位，与上界的大小关系为 p，目前数的大小模 2520 的余数为 j，数位出现的状态数为 k 的方案数。每 dp 到新的一位，只需要枚举之前的状态和这一位选用的数，就可做到 O(1) 转移。</p>
<p>类似地，我们观察维护的最后一维。是否有和之前优化过程类似的冗余的地方？</p>
<p>如果数位当中已经出现了 9，那么出现 3 并不会在这个数上增加什么限制。</p>
<p>题目当中的约束可以等价地转换为：出现了的数位的 LCM |&nbsp;原数%2520。</p>
<p>这些 LCM 只会取到 2520 的因子。2520=2<sup>3</sup>&times;3<sup>2</sup>&times;5&times;7，所以 1 到 9 这九个数的最小公倍数只会出现 4&times;3&times;2&times;2=48 种情况，是可以接受的。状态数降到了18*2*2520*48&asymp;4<sup>6</sup>。</p>
<div class="cnblogs_code">
<pre>#include&lt;bits/stdc++.h&gt;
<span style="color: #0000ff;">#define</span> int long long
<span style="color: #0000ff;">using</span> <span style="color: #0000ff;">namespace</span><span style="color: #000000;"> std;
</span><span style="color: #0000ff;">const</span> <span style="color: #0000ff;">int</span> N=<span style="color: #800080;">2520</span><span style="color: #000000;">;
</span><span style="color: #0000ff;">int</span> t,l,r,cnt,a[<span style="color: #800080;">20</span>],v[N+<span style="color: #800080;">5</span>],f[<span style="color: #800080;">20</span>][N+<span style="color: #800080;">5</span>][<span style="color: #800080;">50</span><span style="color: #000000;">];
</span><span style="color: #0000ff;">int</span> dfs(<span style="color: #0000ff;">int</span> i,<span style="color: #0000ff;">int</span> p,<span style="color: #0000ff;">int</span> x,<span style="color: #0000ff;">bool</span> less){    <span style="color: #008000;">//</span><span style="color: #008000;">i:数位  p:目前数的大小模 2520 的余数  x:i位之前的数的每一位数的最小公倍数</span>
    <span style="color: #0000ff;">if</span>(!i) <span style="color: #0000ff;">return</span> p%x==<span style="color: #800080;">0</span><span style="color: #000000;">;
    </span><span style="color: #0000ff;">if</span>(!less&amp;&amp;f[i][p][v[x]]!=-<span style="color: #800080;">1</span>) <span style="color: #0000ff;">return</span><span style="color: #000000;"> f[i][p][v[x]];
    </span><span style="color: #0000ff;">int</span> end=less?a[i]:<span style="color: #800080;">9</span>,ans=<span style="color: #800080;">0</span><span style="color: #000000;">;
    </span><span style="color: #0000ff;">for</span>(<span style="color: #0000ff;">int</span> j=<span style="color: #800080;">0</span>;j&lt;=end;j++)    <span style="color: #008000;">//</span><span style="color: #008000;">当前数位 </span>
        ans+=dfs(i-<span style="color: #800080;">1</span>,(p*<span style="color: #800080;">10</span>+j)%N,j?x/__gcd(x,j)*j:x,less&amp;(j==end));    <span style="color: #008000;">//</span><span style="color: #008000;">x/__gcd(x,j)*j 即 x*j/gcd(x,j)=lcm(x,j)。计算的是包含当前位时所有位上的数的最小公倍数（当前位所选数不为 0，如果为 0 就是原数） </span>
    <span style="color: #0000ff;">if</span>(!less) f[i][p][v[x]]=<span style="color: #000000;">ans;
    </span><span style="color: #0000ff;">return</span><span style="color: #000000;"> ans;
}
</span><span style="color: #0000ff;">int</span> calc(<span style="color: #0000ff;">int</span><span style="color: #000000;"> x){
    </span><span style="color: #0000ff;">int</span> n=<span style="color: #800080;">0</span><span style="color: #000000;">;
    </span><span style="color: #0000ff;">while</span>(x) a[++n]=x%<span style="color: #800080;">10</span>,x/=<span style="color: #800080;">10</span><span style="color: #000000;">;
    </span><span style="color: #0000ff;">return</span> dfs(n,<span style="color: #800080;">0</span>,<span style="color: #800080;">1</span>,<span style="color: #800080;">1</span><span style="color: #000000;">);
}
signed main(){
    memset(f,</span>-<span style="color: #800080;">1</span>,<span style="color: #0000ff;">sizeof</span><span style="color: #000000;">(f));
    </span><span style="color: #0000ff;">for</span>(<span style="color: #0000ff;">int</span> i=<span style="color: #800080;">1</span>;i&lt;=N;i++<span style="color: #000000;">)
        </span><span style="color: #0000ff;">if</span>(N%i==<span style="color: #800080;">0</span>) v[i]=++cnt;    <span style="color: #008000;">//</span><span style="color: #008000;">标记 2520 的因子 </span>
    scanf(<span style="color: #800000;">"</span><span style="color: #800000;">%lld</span><span style="color: #800000;">"</span>,&amp;<span style="color: #000000;">t);
    </span><span style="color: #0000ff;">while</span>(t--<span style="color: #000000;">){
        scanf(</span><span style="color: #800000;">"</span><span style="color: #800000;">%lld%lld</span><span style="color: #800000;">"</span>,&amp;l,&amp;<span style="color: #000000;">r);
        printf(</span><span style="color: #800000;">"</span><span style="color: #800000;">%lld\n</span><span style="color: #800000;">"</span>,calc(r)-calc(l-<span style="color: #800080;">1</span><span style="color: #000000;">));
    }
    </span><span style="color: #0000ff;">return</span> <span style="color: #800080;">0</span><span style="color: #000000;">;
}</span></pre>
</div>
<h4>6.&nbsp;2012 Multi-University Training Contest 6&nbsp;XHXJ's LIS（HDU 4352）</h4>
<p><strong>题目大意</strong>：给定两个正整数 a 和 b，求在 [a,b] 中的所有整数中，有多少个数（不包含前导零）的数位最长上升子序列长度恰好为 k。T&le;10<sup>4</sup>，a&le;b&le;10<sup>18</sup>。</p>
<p><strong>Solution：</strong></p>
<p>由于 LIS 只能 dp 求，所以我们维护的状态也应是dp状态。</p>
<p>考虑 n<sup>2</sup> LIS 的求法：当确定了一个新的数位后，这个位置的 dp 值跟之前所有位置的dp 值都有关系。维护这些信息的复杂度比维护具体这个数是什么还要高，显然不现实。</p>
<p>1.考虑树状数组优化 LIS 的求法：在树状数组 LIS 的求法中，最终的答案和当新加入一个数后 dp 值的更新只跟树状数组有关。</p>
<ul>
<li>下标：0~9</li>
<li>取值范围：1~10</li>
<li>状态数：10<sup>10</sup>，难以接受！</li>
</ul>
<p>如果我们直接维护树状数组所维护的序列，发现一个性质：单调不降。且由于要求的是严格上升序列，所以 0 位置的值必定不超过 1。</p>
<p>我们考虑维护差分序列，下标范围：0~9， 取值范围：0~1。状态数下降到 2<sup>10</sup>。</p>
<p>2.考虑二分求 LIS 的做法：其中 f(i) 表示长度为 i 的上升子序列结尾最小可以是多少。下标范围：1~10，取值范围：0~9，状态数：10<sup>10</sup>。</p>
<p>冷静思考，发现 f(i) 实际上是严格单增的。</p>
<p>因此只需要记录一下 0~9 哪些数在 f(i) 中出现过就足够还原出整个 dp 数组的信息。状态数 2<sup>10</sup>。</p>
<p>用十位二进制表示 0~9 出现的情况，和二分求 LIS 一样的方法进行更新。</p>
<div class="cnblogs_code">
<pre>#include&lt;bits/stdc++.h&gt;
<span style="color: #0000ff;">#define</span> int long long
<span style="color: #0000ff;">using</span> <span style="color: #0000ff;">namespace</span><span style="color: #000000;"> std;
</span><span style="color: #0000ff;">const</span> <span style="color: #0000ff;">int</span> N=<span style="color: #800080;">11</span><span style="color: #000000;">;
</span><span style="color: #0000ff;">int</span> t,l,r,k,f[<span style="color: #800080;">25</span>][<span style="color: #800080;">1</span>&lt;&lt;N][N][<span style="color: #800080;">2</span>],a[N];    <span style="color: #008000;">//</span><span style="color: #008000;">f[i][j][k][0/1]:当前的数位为 i，状态为 j，要求的 LIS 长度为 k，与上界之间的关系为 0/1 的方案数。 </span>
<span style="color: #0000ff;">int</span> solve(<span style="color: #0000ff;">int</span> x,<span style="color: #0000ff;">int</span> S){    <span style="color: #008000;">//</span><span style="color: #008000;">获取新的状态 </span>
    <span style="color: #0000ff;">for</span>(<span style="color: #0000ff;">int</span> i=x;i&lt;=<span style="color: #800080;">9</span>;i++<span style="color: #000000;">)
        </span><span style="color: #0000ff;">if</span>(S&amp;(<span style="color: #800080;">1</span>&lt;&lt;i)) <span style="color: #0000ff;">return</span> S^(<span style="color: #800080;">1</span>&lt;&lt;i)^(<span style="color: #800080;">1</span>&lt;&lt;<span style="color: #000000;">x);
    </span><span style="color: #0000ff;">return</span> S^(<span style="color: #800080;">1</span>&lt;&lt;<span style="color: #000000;">x);
}
</span><span style="color: #0000ff;">int</span> dfs(<span style="color: #0000ff;">int</span> i,<span style="color: #0000ff;">int</span> S,<span style="color: #0000ff;">bool</span> have0,<span style="color: #0000ff;">bool</span> less){    <span style="color: #008000;">//</span><span style="color: #008000;">i:数位  S:当前的状态  have0:是否含有前导零  less:与上界之间的关系 </span>
    <span style="color: #0000ff;">if</span>(!i) <span style="color: #0000ff;">return</span> __builtin_popcount(S)==k;    <span style="color: #008000;">//</span><span style="color: #008000;">状态中 1 的数目就是 LIS 的长度 </span>
    <span style="color: #0000ff;">if</span>(!less&amp;&amp;f[i][S][k][less]!=-<span style="color: #800080;">1</span>) <span style="color: #0000ff;">return</span><span style="color: #000000;"> f[i][S][k][less];
    </span><span style="color: #0000ff;">int</span> end=less?a[i]:<span style="color: #800080;">9</span>,ans=<span style="color: #800080;">0</span><span style="color: #000000;">;
    </span><span style="color: #0000ff;">for</span>(<span style="color: #0000ff;">int</span> j=<span style="color: #800080;">0</span>;j&lt;=end;j++)    <span style="color: #008000;">//</span><span style="color: #008000;">当前数位 </span>
        ans+=dfs(i-<span style="color: #800080;">1</span>,(have0&amp;&amp;j==<span style="color: #800080;">0</span>)?<span style="color: #800080;">0</span>:solve(j,S),have0&amp;&amp;j==<span style="color: #800080;">0</span>,less&amp;&amp;j==<span style="color: #000000;">end);
    </span><span style="color: #0000ff;">if</span>(!less) f[i][S][k][less]=<span style="color: #000000;">ans;
    </span><span style="color: #0000ff;">return</span><span style="color: #000000;"> ans;
}
</span><span style="color: #0000ff;">int</span> calc(<span style="color: #0000ff;">int</span><span style="color: #000000;"> x){
    </span><span style="color: #0000ff;">int</span> n=<span style="color: #800080;">0</span><span style="color: #000000;">;
    </span><span style="color: #0000ff;">while</span>(x) a[++n]=x%<span style="color: #800080;">10</span>,x/=<span style="color: #800080;">10</span><span style="color: #000000;">;
    </span><span style="color: #0000ff;">return</span> dfs(n,<span style="color: #800080;">0</span>,<span style="color: #800080;">1</span>,<span style="color: #800080;">1</span><span style="color: #000000;">);
}
signed main(){
    memset(f,</span>-<span style="color: #800080;">1</span>,<span style="color: #0000ff;">sizeof</span><span style="color: #000000;">(f));
    scanf(</span><span style="color: #800000;">"</span><span style="color: #800000;">%lld</span><span style="color: #800000;">"</span>,&amp;<span style="color: #000000;">t);
    </span><span style="color: #0000ff;">for</span>(<span style="color: #0000ff;">int</span> i=<span style="color: #800080;">1</span>;i&lt;=t;i++<span style="color: #000000;">){
        scanf(</span><span style="color: #800000;">"</span><span style="color: #800000;">%lld%lld%lld</span><span style="color: #800000;">"</span>,&amp;l,&amp;r,&amp;<span style="color: #000000;">k);
        printf(</span><span style="color: #800000;">"</span><span style="color: #800000;">Case #%lld: %lld\n</span><span style="color: #800000;">"</span>,i,calc(r)-calc(l-<span style="color: #800080;">1</span><span style="color: #000000;">));
    }
    </span><span style="color: #0000ff;">return</span> <span style="color: #800080;">0</span><span style="color: #000000;">;
}</span></pre>
</div>
<h4>7.HDU4507&nbsp;恨7不成妻</h4>
<p><strong>题目大意：</strong>如果一个整数符合下面 3 个条件之一，那么我们就说这个整数和 7 有关&mdash;&mdash;</p>
<ul>
<li>整数中某一位是 7；</li>
<li>整数的每一位加起来的和是 7 的整数倍；</li>
<li>这个整数是 7 的整数倍；</li>
</ul>
<p>求在区间 [L,R] 内和 7 无关的数字的平方和。L,R&le;10<sup>18</sup>。</p>
<p><strong>Solution：</strong></p>
<p>&ldquo;某一位是 7&rdquo;只需记录是否出现过 7，&ldquo;每一位加起来的和是 7 的整数倍&rdquo;只需记录数位和 mod 7 的值，&ldquo;是 7 的整数倍&rdquo;只需记录当前数 mod 7 的值。此题的重点在于如何在 dp 的过程中维护平方和。</p>
<p>我们之前接触的大部分数位 dp 统计的都是满足性质的数的个数。转移时，都是若满足条件，则转移后的 dp 值+=转移前的 dp 值。思考这个过程。</p>
<p>令 f<sub>0</sub> 表示满足性质的数的个数。</p>
<p>转移前的 f<sub>0</sub>：&Sigma;<sub>合法的 xi&nbsp;</sub>1</p>
<p><span class="mrow"><span class="mo">假设当前的数位是 c。在每一个数的后面都加上当前的数位 c，并不会改变它们的个数（一个数后面添加一个 c 仍是一个数）。</span></span></p>
<p><span class="mrow"><span class="mo">所以转移后的 f<sub>0</sub> 依然是：&Sigma;<sub>合法的 xi&nbsp;</sub>1</span></span></p>
<p><span class="mrow"><span class="mo">&Sigma;<sub>合法的 xi&nbsp;</sub>1<span id="MathJax-Span-41" class="mrow"><span id="MathJax-Span-42" class="mo">&rArr;&Sigma;<sub>合法的 xi&nbsp;</sub>1，转移前和转移后的 f<sub>0</sub> 都是满足性质的数的个数，这也就是为什么可以直接把转移前的 dp 值加到转移后的 dp 值上。</span></span></span></span></p>
<p><span class="mrow"><span class="mo"><span class="mrow"><span class="mo">思考另一个东西。</span></span></span></span></p>
<p><span class="mrow"><span class="mo">令 f<sub>1</sub> 表示满足条件的数之和。</span></span></p>
<p><span class="mrow"><span class="mo">&Sigma;<sub>合法的 xi&nbsp;</sub>x<sub>i</sub>&rArr;&Sigma;<sub>合法的 xi</sub> (10x<sub>i</sub>+c)</span></span>&rArr;10 &Sigma;<sub>合法的 xi</sub> x<sub>i</sub>+c&middot;&Sigma;<sub>合法的 xi</sub> 1&rArr;10&middot;转移前的 f<sub>1</sub>+c&middot;转移前的 f<sub>0</sub></p>
<p>如果我们维护了 f<sub>0</sub>，那么就可以通过这个式子去更新 f<sub>1</sub>。</p>
<p>类似地，令 f<sub>2</sub> 表示满足条件的数的平方和。</p>
<p>&Sigma;<sub>合法的 xi</sub>&nbsp;x<sub>i</sub><sup>2</sup>&rArr;&Sigma;<sub>合法的 xi</sub>&nbsp;(10x<sub>i</sub>+c)<sup>2</sup>&rArr;100&Sigma;<sub>合法的 xi</sub> x<sub>i</sub><sup>2</sup>+20&middot;c&middot;&Sigma;<sub>合法的 xi</sub> x<sub>i</sub>+c<sup>2</sup>&middot;&Sigma;<sub>合法的 xi</sub> 1&rArr;100&middot;转移前的 f<sub>2</sub>+20&middot;c&middot;转移前的&nbsp;f<sub>1</sub>+c<sup>2</sup>&middot;转移前的 f<sub>0</sub></p>
<p>于是，如果我们已经维护了 f<sub>0</sub> 和 f<sub>1</sub>，就可以完成对 f<sub>2</sub> 的维护。</p>

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

