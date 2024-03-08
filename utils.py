def get_product_cost(products, product_costs, class_id, track_id):
    product_data = products.get(class_id, {}).get(track_id, {'tracked': False})
    return product_data['tracked'] * product_costs.get(class_id, 0)