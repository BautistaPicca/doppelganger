import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Star, Trophy } from "lucide-react";

const categories = [
  { id: "all", label: "Todos los famosos", icon: Star },
  { id: "soccer", label: "Fútbolistas", icon: Trophy },
];

interface CategoryFilterProps {
  selectedCategory: string;
  onCategoryChange: (category: string) => void;
}

export const CategoryFilter = ({ selectedCategory, onCategoryChange }: CategoryFilterProps) => {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-center">Seleccionar categoría</CardTitle>
        <p className="text-sm text-muted-foreground text-center">
          Selecciona una categoría para filtrar los famosos disponibles
        </p>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          {categories.map((category, index) => {
            const Icon = category.icon;
            const isSelected = selectedCategory === category.id;
            
            return (
              <motion.div
                key={category.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1, duration: 0.3 }}
              >
                <Button
                  variant={isSelected ? "default" : "outline"}
                  size="lg"
                  className={`h-20 w-full flex-col gap-2 text-xs transition-all duration-200`}
                  onClick={() => onCategoryChange(category.id)}
                >
                  <Icon className="h-5 w-5" />
                  <span className="font-medium">{category.label}</span>
                </Button>
              </motion.div>
            );
          })}
        </div>

        {selectedCategory !== "all" && (
          <motion.div 
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex justify-center mt-4"
          >
            <Badge variant="secondary">
              {categories.find(c => c.id === selectedCategory)?.label} fue seleccionado
            </Badge>
          </motion.div>
        )}
      </CardContent>
    </Card>
  );
};