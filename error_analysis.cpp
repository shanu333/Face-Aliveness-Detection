#include <iostream>
#include <cstdio>

using namespace std;

int main()
{
    int c = 1, w = 1;
    char a, b, c1, d;
    freopen("4.txt", "r", stdin);
    freopen("final1.txt", "a", stdout);
    cin >> a >> b;
    while (cin >> c1 >> d) {
        w++;
        if (a == c1 && d == b) {
            c++;
        }
    }
    cout << (double)c / (double)w * (double)100 << endl;
}
