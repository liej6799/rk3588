#!/usr/bin/env python3
import os
import sys

support = [
  
]

import clang.cindex as ci
ci.Config.set_library_file("/usr/lib/aarch64-linux-gnu/libclang-14.so.1")
index = ci.Index.create()

lookup = {}
def walk(node):
  if node.kind == ci.CursorKind.TYPEDEF_DECL:
    lookup[node.spelling] = node
  for n in node.get_children():
    walk(n)

def nprint(node, pt=""):
  #if node.kind == ci.CursorKind.FIELD_DECL:
  print(pt, node.kind, node.spelling, node.location.file, node.location.line, node.location.column)
  for n in node.get_children():
    nprint(n, " "+pt)

typeref_to_printf = {

}

def print_field(field):
  ll = 30
  if field.kind == ci.CursorKind.UNION_DECL:
    print(f'  printf("%{ll}s: <union> ", "{field.spelling}");')
    return
  if field.kind != ci.CursorKind.FIELD_DECL:
    print(field.kind, file=sys.stderr)
  assert field.kind == ci.CursorKind.FIELD_DECL
  typeref = None
  is_array = False
  for n in field.get_children():
    if n.kind == ci.CursorKind.TYPE_REF:
      typeref = n.spelling
    if n.kind == ci.CursorKind.INTEGER_LITERAL:
      is_array = True
  #print(typeref, field.spelling)
  if is_array:
    print(f'  printf("%{ll}s: <{typeref} is array> ", "{field.spelling}");')
  else:
    if typeref in typeref_to_printf:
      print(f'  printf("%{ll}s: {typeref_to_printf[typeref]} ", "{field.spelling}", p->{field.spelling});')
    else:
      print(f'  printf("%{ll}s: <{typeref} not parsed> ", "{field.spelling}");')

tu = index.parse("include.cc",
  args = [
    "-I../rknn-header"
  ])

walk(tu.cursor)
for s in support:
  print("// ******", s)
  #nprint(lookup[s])
  print(f"void pprint({s} *p)", "{")
  #print(f'  return;')
  print(f'  printf("    {s} ");')
  i = 0
  for x in list(lookup[s].get_children())[0].get_children():
    #if i%4 == 0 and i!=0:
    print('  printf("\\n");')
    print_field(x)
    i += 1
  print('  printf("\\n");')
  print("}")

exit(0)